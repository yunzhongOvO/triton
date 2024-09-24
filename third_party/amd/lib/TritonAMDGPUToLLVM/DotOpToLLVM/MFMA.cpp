/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "../PatternTritonGPUOpToLLVM.h"
#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "Utility.h"

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

using ::mlir::LLVM::AMD::shuffleXor;
using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

using ValueTable = std::map<std::array<int, 3>, llvm::SmallVector<Value>>;

struct DotOpMFMAConversionHelper {
  AMDMfmaEncodingAttr mfmaLayout;

  ConversionPatternRewriter &rewriter;
  const LLVMTypeConverter *typeConverter;
  Location loc;
  MLIRContext *ctx{};

  explicit DotOpMFMAConversionHelper(AMDMfmaEncodingAttr mfmaLayout,
                                     ConversionPatternRewriter &rewriter,
                                     const LLVMTypeConverter *typeConverter,
                                     Location loc)
      : mfmaLayout(mfmaLayout), rewriter(rewriter),
        typeConverter(typeConverter), loc(loc), ctx(mfmaLayout.getContext()) {}

  Value getThreadId() const {
    auto llvmIndexTy = typeConverter->getIndexType();
    auto tid = rewriter.create<::mlir::gpu::ThreadIdOp>(
        loc, rewriter.getIndexType(), ::mlir::gpu::Dimension::x);
    return rewriter.create<arith::TruncIOp>(loc, i32_ty, tid);
  }

  /**
    * @param mfmaInstnName
    * @param valA
    * @param valB
    * @param valC
    * @param cbsz Control Broadcast Size modifier
    * @param abid A-matrix Broadcast Identifier
    * @param blgp B-matrix Lane Group Pattern modifier
  */
  Value generateMFMAOp(StringRef mfmaInsnName, Value valA, Value valB,
                       Value valC, int cbsz = 0, int abid = 0,
                       int blgp = 0) const {
    assert(cbsz >= 0 && cbsz <= 4);
    assert(abid >= 0 && abid <= 15);
    assert(blgp >= 0 && blgp <= 7);
    auto resType = valC.getType();
    Value zeroFlag = i32_val(0);
    Value cbszFlag = cbsz != 0 ? i32_val(cbsz) : zeroFlag;
    Value abidFlag = abid != 0 ? i32_val(abid) : zeroFlag;
    Value blgpFlag = blgp != 0 ? i32_val(blgp) : zeroFlag;
    OperationState loweredOp(loc, mfmaInsnName);
    loweredOp.addTypes(resType);
    loweredOp.addOperands({valA, valB, valC, cbszFlag, abidFlag, blgpFlag});
    return rewriter.create(loweredOp)->getResult(0);
  }

  Value broadcastGroup(Value val, int groupId, int numGroups) const {
    constexpr int waveSize = 64;
    const int groupSize = waveSize / numGroups;

    Value lane = getThreadId();
    // Multiply by 4, because permute requires offset in bytes
    Value laneOffset = mul(urem(lane, i32_val(groupSize)), i32_val(4));
    Value permuteAddr = add(laneOffset, i32_val(groupId * groupSize * 4));
    Type valType = val.getType();
    Value broadcasted;
    if (valType.isInteger(32))
      broadcasted = rewriter.create<ROCDL::DsBpermuteOp>(loc, val.getType(), permuteAddr, val);
    if (valType.isF32()) {
      val = bitcast(val, i32_ty);
      broadcasted = rewriter.create<ROCDL::DsBpermuteOp>(loc, val.getType(), permuteAddr, val);
      broadcasted = bitcast(broadcasted, f32_ty);
    }
    if (mlir::dyn_cast<VectorType>(valType)) {
    // if (valType.isa<VectorType>) {
      auto vecTy = mlir::dyn_cast<VectorType>(valType);
      auto vecBitSize = vecTy.getElementType().getIntOrFloatBitWidth() * vecTy.getNumElements();
      const int int32VecSize = vecBitSize / 32;

      Type int32VecTy = vec_ty(i32_ty, int32VecSize);
      Value int32Val = bitcast(val, int32VecTy);
      Value int32Broadcasted = undef(int32VecTy);
      for (int i = 0; i < int32VecSize; ++i) {
        Value int32Chunk = extract_element(i32_ty, int32Val, i32_val(i));
        Value broadcastedChunk = rewriter.create<ROCDL::DsBpermuteOp>(loc, i32_ty, permuteAddr, int32Chunk);
        int32Broadcasted = insert_element(int32VecTy, int32Broadcasted, broadcastedChunk, i32_val(i));
      }
      broadcasted = bitcast(int32Broadcasted, valType);
    }
    assert(broadcasted);
    return broadcasted;
  }

  Value rollShiftOperand(Value val) const {
    constexpr int waveSize = 64;
    constexpr int numGroups = 16;
    const int groupSize = waveSize / numGroups;

    Value lane = getThreadId();
    Value targetLane = urem(add(lane, i32_val(groupSize)), i32_val(waveSize));

    // Multiply by 4, because permute requires offset in bytes
    Value permuteAddr = mul(targetLane, i32_val(4));

    Type valType = val.getType();
    Value rolled;
    if (valType.isInteger(32))
      rolled = rewriter.create<ROCDL::DsBpermuteOp>(loc, valType, permuteAddr, val);
    
    if (valType.isF32()) {
      val = bitcast(val, i32_ty);
      rolled = rewriter.create<ROCDL::DsBpermuteOp>(loc, valType, permuteAddr, val);
      rolled = bitcast(rolled, f32_ty);
    }
    if (mlir::dyn_cast<VectorType>(valType)) {
      auto vecTy = mlir::dyn_cast<VectorType>(valType);
      // auto vecTy = valType.dyn_cast<VectorType>();
      auto vecBitSize = vecTy.getElementType().getIntOrFloatBitWidth() * vecTy.getNumElements();
      const int int32VecSize = vecBitSize / 32;

      Type int32VecTy = vec_ty(i32_ty, int32VecSize);
      Value int32Val = bitcast(val, int32VecTy);
      Value int32Rolled = undef(int32VecTy);
      for (int i = 0; i < int32VecSize; ++i) {
        Value int32Chunk = extract_element(i32_ty, int32Val, i32_val(i));
        Value rolledChunk = rewriter.create<ROCDL::DsBpermuteOp>(loc, i32_ty, permuteAddr, int32Chunk);
        int32Rolled = insert_element(int32VecTy, int32Rolled, rolledChunk, i32_val(i));
      }
      rolled = bitcast(int32Rolled, valType);
    }
    assert(rolled);
    return rolled;
  }

  Value generateMFMATile(StringRef mfmaInsnName, SmallVector<Value> valA,
                         SmallVector<Value> valB, Value valC, int mDim,
                         int nDim, bool transpose) const {
    Value acc;
    if (mDim == nDim) {
      assert(valA.size() == 1 && valB.size() == 1);
      acc = transpose ? generateMFMAOp(mfmaInsnName, valB[0], valA[0], valC):
                        generateMFMAOp(mfmaInsnName, valA[0], valB[0], valC);
    }
    if (mDim == 4 && nDim == 64 || mDim == 64 && nDim == 4) {
      // broadcast selected kRep A operand matrix to all A matrices(2^4=16)
      constexpr int broadcastCtrl = 4;
      constexpr int numRepeats = 16;
      acc = valC;
      for (int kRep = 0; kRep < numRepeats; kRep++) {
        if (mDim == 4) {
          assert(valA.size() == 1 && valB.size() == 16);
          if (!transpose)
            acc = generateMFMAOp(mfmaInsnName, valA[0], valB[kRep], acc);
          else
            acc = generateMFMAOp(mfmaInsnName, valA[kRep], valB[0], acc);
          valA[0] = rollShiftOperand(valA[0]);
        }
        if (nDim == 4) {
          assert(valA.size() == 16 && valB.size() == 1);
          if (!transpose)
            acc = generateMFMAOp(mfmaInsnName, valA[kRep], valB[0], acc);
          else
            acc = generateMFMAOp(mfmaInsnName, valB[0], valA[kRep], acc);
          valB[0] = rollShiftOperand(valB[0]);
        }
      }
    }
    return acc;
  }

  int getNumSubmatrices(Type elementType, int mDim, int nDim) const {
    if ((mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64))
      return 1;
    assert(mDim == nDim);
    switch (mDim) {
    case 32:
    case 16:
      return 1;
      break;
    case 4:
      assert(elementType.getIntOrFloatBitWidth() <= 32 &&
             "fp64 is not supported yet");
      assert(elementType.getIntOrFloatBitWidth() != 8 ||
             elementType.isInteger(8) && "fp8 is not supported yet");
      return 16;
      break;
    default:
      llvm::report_fatal_error("unsupported nonKDim in MFMA dot");
    }
    return -1;
  }

  Value processSubBlocks(int numSubBlocks, Value acc, bool reduceSubBlocks,
                         bool zeroSubBlocks) const {
    assert((numSubBlocks & (numSubBlocks - 1)) == 0 &&
           "numSubBlocks in not pow 2!");
    if (numSubBlocks == 1)
      return acc;
    constexpr int warpSize = 64;
    int subBlockSize = warpSize / numSubBlocks;
    Value laneId = getThreadId();
    laneId = and_(laneId, i32_val(warpSize - 1));
    auto vecTy = dyn_cast<VectorType>(acc.getType());
    auto elemType = vecTy.getElementType();
    assert(elemType.getIntOrFloatBitWidth() == 32);
    int numScalars = vecTy.getNumElements();
    std::vector<Value> accScalar(numScalars);
    for (int i = 0; i < numScalars; ++i)
      accScalar[i] = extract_element(elemType, acc, i32_val(i));

    if (reduceSubBlocks) {
      while (subBlockSize < warpSize) {
        for (int i = 0; i < numScalars; ++i) {
          Value other_acc =
              shuffleXor(loc, rewriter, accScalar[i], subBlockSize);
          if (elemType.isInteger(32))
            accScalar[i] = add(accScalar[i], other_acc);
          else
            accScalar[i] = fadd(accScalar[i], other_acc);
        }
        subBlockSize *= 2;
      }
    }
    if (zeroSubBlocks) {
      Value zero;
      if (elemType.isInteger(32))
        zero = i32_val(0);
      else
        zero = f32_val(0.0);
      auto cond = icmp_ult(laneId, i32_val(subBlockSize));
      for (int i = 0; i < numScalars; ++i)
        accScalar[i] = select(cond, accScalar[i], zero);
    }

    Value reducedAcc = undef(vecTy);
    for (int i = 0; i < numScalars; ++i)
      reducedAcc = insert_element(vecTy, reducedAcc, accScalar[i], i32_val(i));
    return reducedAcc;
  }

  /// @brief MFMA 4x4 is computes 16 matrix multiplications, this functions adds
  /// these 16 matrices to get final 4x4 matrix
  /// @param numSubBlocks
  /// @param acc
  /// @return
  Value reduceSubBlocks(int numSubBlocks, Value acc) const {
    return processSubBlocks(numSubBlocks, acc, true, false);
  }

  /// @brief Zeroes out redundant values in all sub-blocks except first one
  ///
  /// Every warp in mfma 4x4 layout holds only 4 unique values(scalar or
  /// vectors) in blocks of 4 consecutive threads, There are 16 copies of these
  /// 4 values across all threads of the warp. Need to zero out 15 copies to use
  /// accumulator between dot operations.
  /// @param numSubBlocks
  /// @param acc
  /// @return
  Value zeroAuxiliarBlocks(int numSubBlocks, Value acc) const {
    return processSubBlocks(numSubBlocks, acc, false, true);
  }

  // Conduct the Dot conversion.
  LogicalResult convertDot(DotOp op, DotOpAdaptor adaptor) const {
    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();
    auto mDim = mfmaLayout.getMDim();
    auto nDim = mfmaLayout.getNDim();
    auto mfmaVersion = mfmaLayout.getVersionMajor();
    assert((mDim == nDim && (mDim == 32 || mDim == 16 || mDim == 4)) ||
           (mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64));

    Value a = op.getA();
    Value b = op.getB();
    Value d = op.getD();
    auto aTensorTy = cast<RankedTensorType>(a.getType());
    auto bTensorTy = cast<RankedTensorType>(b.getType());
    auto dTensorTy = cast<RankedTensorType>(d.getType());
    auto elemTyA = aTensorTy.getElementType();
    auto elemTyB = bTensorTy.getElementType();

    StringRef mfmaInsnName;
    auto maybeMfmaInsn =
        MfmaInsn::selectMfma(mDim, nDim, elemTyA, elemTyB, mfmaVersion);
    if (failed(maybeMfmaInsn))
      llvm::report_fatal_error("No match found in MFMA database\n");

    mfmaInsnName = maybeMfmaInsn->getInsnName();
    // unsigned kBase = maybeMfmaInsn->getKBase();
    unsigned kBaseA = maybeMfmaInsn->getKBaseA();
    unsigned kBaseB = maybeMfmaInsn->getKBaseB();

    auto aEncoding = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
    auto bEncoding = cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());

    // int kWidth = aEncoding.getKWidth();
    int kWidthA = aEncoding.getKWidth();
    int kWidthB = bEncoding.getKWidth();

    auto rank = aTensorTy.getShape().size();

    auto repA =
        mfmaLayout.getMFMARepForOperands(aTensorTy.getShape(), kWidthA, 0);
    auto repB =
        mfmaLayout.getMFMARepForOperands(bTensorTy.getShape(), kWidthB, 1);

    assert(repA[2] == repB[1]);

    Value loadedA = adaptor.getA();
    Value loadedB = adaptor.getB();
    Value loadedC = adaptor.getC();

    auto numRepM = repA[1];
    auto numRepN = repB[2];
    auto numRepK = repA[2];
    auto numRepB = repA[0];
    assert(repA[0] == repB[0]);

    auto operandA = getValuesFromDotOperandLayoutStruct(
        loadedA, numRepB, numRepM, numRepK, kWidthA, kBaseA,
        aTensorTy.getElementType());
    auto operandB = getValuesFromDotOperandLayoutStruct(
        loadedB, numRepB, numRepN, numRepK, kWidthB, kBaseB,
        aTensorTy.getElementType());

    auto dstElemTy = dTensorTy.getElementType();
    auto fc = unpackLLElements(loc, loadedC, rewriter);

    unsigned warpSize = triton::gpu::getWarpSize(mfmaLayout);
    // compute number of output elements that each thread holds for one MFMA
    // instruction. subBlocks
    const int subBlocks =
        getNumSubmatrices(aTensorTy.getElementType(), mDim, nDim);
    auto elemsPerVec = mDim * nDim * subBlocks / warpSize;

    auto vecTy = vec_ty(dstElemTy, elemsPerVec);
    for (int b = 0; b < numRepB; ++b) {
      for (int m = 0; m < numRepM; ++m) {
        for (int n = 0; n < numRepN; ++n) {
          Value acc = undef(vecTy);
          for (unsigned v = 0; v < elemsPerVec; ++v) {
            acc = insert_element(
                vecTy, acc,
                fc[b * numRepM * numRepN * elemsPerVec +
                   m * numRepN * elemsPerVec + n * elemsPerVec + v],
                i32_val(v));
          }
          acc = zeroAuxiliarBlocks(subBlocks, acc);
          for (int k = 0; k < numRepK; k++) {
            for (int kPack = 0; kPack < kWidthA / kBaseA; ++kPack)
              // acc =
              //     mfmaLayout.getIsTransposed()
              //         ? generateMFMAOp(mfmaInsnName, operandB[kPack][{b, n, k}],
              //                          operandA[kPack][{b, m, k}], acc)
              //         : generateMFMAOp(mfmaInsnName, operandA[kPack][{b, m, k}],
              //                          operandB[kPack][{b, n, k}], acc);
              acc = generateMFMATile(mfmaInsnName, operandA[kPack][{b, m, k}],
                                     operandB[kPack][{b, n, k}], acc, mDim, nDim,
                                     mfmaLayout.getIsTransposed());
          }
          acc = reduceSubBlocks(subBlocks, acc);
          for (unsigned v = 0; v < elemsPerVec; ++v) {
            fc[b * numRepM * numRepN * elemsPerVec + m * numRepN * elemsPerVec +
               n * elemsPerVec + v] =
                extract_element(dstElemTy, acc, i32_val(v));
          }
        }
      }
    }
    // replace with new packed result
    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(fc.size(), dstElemTy));
    Value res = packLLElements(loc, typeConverter, fc, rewriter, structTy);
    rewriter.replaceOp(op, res);

    return success();
  }

  /**
   * @brief extract vector from rawElems based on kWidth and kBase
   * rawElems is a vector of kWidth elements. We need to prepare vector(s) of
   * kBase elements for each mfma instruction
   * 
   * @param rawElems vector of "raw" elements for one mfma tile
   * @param k id in k-pack
   * @param kPack size of k-pack
   * @param numIntrinsics number of operands we need to extract
   * @param type type mfma intrinsic requires
   * 
   * @return elements converted for one repetition
   */
  SmallVector<Value> extractOperands(Value rawElems, int k, int kPack,
                                     int numIntrinsics, Type type) const {
    // int kpack = kWidth / kBase;
    assert(numIntrinsics == 1 || numIntrinsics == 16);
    // auto rawTy = rawElems.getType().cast<VectorType>();
    auto rawTy = mlir::dyn_cast<VectorType>(rawElems.getType());
    auto rawElemTy = rawTy.getElementType();
    // number of elements required by on mfma intrinsic
    int intrinsicK = rawTy.getNumElements() / numIntrinsics / kPack;
    int kBase = rawTy.getNumElements() / kPack;

    SmallVector<Value> results;
    // extracts needed elements in original dtype
    auto typedVecTy = vec_ty(rawElemTy, intrinsicK);
    if (type.isBF16())
      typedVecTy = vec_ty(i16_ty, intrinsicK);
    for (int intrinsic = 0; intrinsic < numIntrinsics; ++intrinsic) {
      Value typedVec = undef(typedVecTy);
      for (int elemId = 0; elemId < intrinsicK; ++elemId) {
        int elemOff = elemId + intrinsic * intrinsicK + k * kBase;
        auto val = extract_element(rawElemTy, rawElems, i32_val(elemOff));
        if (type.isBF16()) {
          // rocdl.mfma.f32.32x32x8bf16.1k calls for input of i16 type
          auto cast = bitcast(val, i16_ty);
          typedVec = insert_element(typedVecTy, typedVec, cast, i32_val(elemId));
        } else
          typedVec = insert_element(typedVecTy, typedVec, val, i32_val(elemId));
      }
      Value castedVec = bitcast(typedVec, type);
      results.push_back(castedVec);
    }
    // auto vecTy = vec_ty(type, kBase);
    // if (type.isBF16())
    //   vecTy = vec_ty(i16_ty, kBase);
    // for (int k = 0; k < kpack; ++k) {
    //   Value vec = undef(vecTy);
    //   for (int elemId = 0; elemId < kBase; ++elemId) {
    //     auto val = extract_element(type, rawElems, i32_val(elemId + k * kBase));
    //     if (type.isBF16()) {
    //       // rocdl.mfma.f32.32x32x8bf16.1k calls for input of i16 type
    //       auto cast = bitcast(val, i16_ty);
    //       vec = insert_element(vecTy, vec, cast, i32_val(elemId));
    //     } else
    //       vec = insert_element(vecTy, vec, val, i32_val(elemId));
    //   }
    //   if (type.getIntOrFloatBitWidth() == 8) {
    //     if (4 == kBase)
    //       // This is for int8 on pre- MI300 GPUs
    //       results.push_back(bitcast(vec, i32_ty));
    //     if (8 == kBase)
    //       results.push_back(bitcast(vec, i64_ty));
    //   } else
    //     results.push_back(vec);
    // }
    return results;
  }

  /**
   * @brief Converts dot operand structure to value table and converts types
   * appropriate for mfma instructions
   */
  SmallVector<ValueTable>
  getValuesFromDotOperandLayoutStruct(Value value, int batch, int n0, int n1,
                                      int kWidth, int kBase, Type type) const {
    auto elems = unpackLLElements(loc, value, rewriter);
    int kpack = kWidth / kBase;
    // "Wide operand" means that this operand id for mfma 4x64 layout
    // This operand is 64x64 for fp16, bf16 and int8 data types and
    // 16x64 for fp32
    bool wideOperand = kWidth >= 16;
    // How many rocdl intrinsics will process one tile
    int numIntrinsics = wideOperand ? 16 : 1;
    int intrinsicKWidth = wideOperand ? kBase / numIntrinsics : kBase;
    Type intrinsicDType;
    if (type.isF32())
      intrinsicDType = f32_ty;
    if (type.getIntOrFloatBitWidth() == 8)
      intrinsicDType = rewriter.getIntegerType(intrinsicKWidth * 8);
    if (type.isBF16())
      intrinsicDType = vec_ty(i16_ty, intrinsicKWidth);
    if (type.isF16())
      intrinsicDType = vec_ty(f16_ty, intrinsicKWidth);
    assert(intrinsicDType);

    SmallVector<ValueTable> dotOpVals(kpack);
    for (int b = 0; b < batch; ++b) {
      for (int i = 0; i < n0; i++) {
        for (int j = 0; j < n1; j++) {
          Type elemTy = typeConverter->convertType(type);
          Type ty = vec_ty(elemTy, kWidth);
          Value rawElems = undef(ty);
          for (int k = 0; k < kWidth; ++k) {
            rawElems = insert_element(
                ty, rawElems,
                elems[kWidth * n1 * n0 * b + kWidth * n1 * i + kWidth * j + k],
                i32_val(k));
          }

          // // Value convertedElems;
          // if (type.isF32()) {
          //   for (int k = 0; k < kpack; ++k)
          //     dotOpVals[k][{b, i, j}] =
          //         extract_element(type, rawElems, i32_val(k));
          // } else {
          //   SmallVector<Value> vals;
          //   if (type.getIntOrFloatBitWidth() == 8) {
          //     vals = extractOperands(rawElems, kWidth, kBase, i8_ty);
          //   } else if (type.isBF16()) {
          //     vals = extractOperands(rawElems, kWidth, kBase, bf16_ty);
          //   } else {
          //     assert(type.isF16() && "Unsupported data type");
          //     vals = extractOperands(rawElems, kWidth, kBase, f16_ty);
          //   }
          for (int k = 0; k < kpack; ++k) {
            SmallVector<Value> vals = extractOperands(
              rawElems, k, kpack, numIntrinsics, intrinsicDType);
            assert(vals.size() == numIntrinsics);
            dotOpVals[k][{b, i, j}] = vals;
          }
          // }
        }
      }
    }
    return dotOpVals;
  }
};

} // namespace

namespace mlir::triton::AMD {
LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) {
  auto rankedTType = [](Value tensor) {
    return cast<RankedTensorType>(tensor.getType());
  };

  assert(isa<DotOperandEncodingAttr>(rankedTType(op.getA()).getEncoding()) &&
         isa<DotOperandEncodingAttr>(rankedTType(op.getB()).getEncoding()) &&
         "Both $a and %b should be DotOperand layout.");

  auto cTensorTy = rankedTType(op.getC());
  auto dTensorTy = rankedTType(op.getD());
  assert(isa<AMDMfmaEncodingAttr>(cTensorTy.getEncoding()) &&
         "Currently, we only support $c with a mfma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's $c operand should pass the same number of values as $d");

  auto loc = op.getLoc();
  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());

  DotOpMFMAConversionHelper helper(mfmaLayout, rewriter, typeConverter, loc);

  return helper.convertDot(op, adaptor);
}
} // namespace mlir::triton::AMD
