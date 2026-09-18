//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2014-2022, NJIT, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================

/*
  Simple examples for CKKS
 */

 #define PROFILE_ALL_MODULI
 // #define PROFILE_ALL_DATA
 #define CROSS_LIBRARY_REFERENCE

 #define DEBUG_PRINT_KEYS
 #include "openfhe.h"
 #include <iostream>

 #include <chrono>

 using namespace lbcrypto;

 int main() {
     // Step 1: Setup CryptoContext

     // A. Specify main parameters
     /* A1) Multiplicative depth:
    * The CKKS scheme we setup here will work for any computation
    * that has a multiplicative depth equal to 'multDepth'.
    * This is the maximum possible depth of a given multiplication,
    * but not the total number of multiplications supported by the
    * scheme.
    *
    * For example, computation f(x, y) = x^2 + x*y + y^2 + x + y has
    * a multiplicative depth of 1, but requires a total of 3 multiplications.
    * On the other hand, computation g(x_i) = x1*x2*x3*x4 can be implemented
    * either as a computation of multiplicative depth 3 as
    * g(x_i) = ((x1*x2)*x3)*x4, or as a computation of multiplicative depth 2
    * as g(x_i) = (x1*x2)*(x3*x4).
    *
    * For performance reasons, it's generally preferable to perform operations
    * in the shorted multiplicative depth possible.
    */
     // uint32_t multDepth = 1;

     /* A2) Bit-length of scaling factor.
    * CKKS works for real numbers, but these numbers are encoded as integers.
    * For instance, real number m=0.01 is encoded as m'=round(m*D), where D is
    * a scheme parameter called scaling factor. Suppose D=1000, then m' is 10 (an
    * integer). Say the result of a computation based on m' is 130, then at
    * decryption, the scaling factor is removed so the user is presented with
    * the real number result of 0.13.
    *
    * Parameter 'scaleModSize' determines the bit-length of the scaling
    * factor D, but not the scaling factor itself. The latter is implementation
    * specific, and it may also vary between ciphertexts in certain versions of
    * CKKS (e.g., in FLEXIBLEAUTO).
    *
    * Choosing 'scaleModSize' depends on the desired accuracy of the
    * computation, as well as the remaining parameters like multDepth or security
    * standard. This is because the remaining parameters determine how much noise
    * will be incurred during the computation (remember CKKS is an approximate
    * scheme that incurs small amounts of noise with every operation). The
    * scaling factor should be large enough to both accommodate this noise and
    * support results that match the desired accuracy.
    */
     // uint32_t scaleModSize = 50;

     /* A3) Number of plaintext slots used in the ciphertext.
    * CKKS packs multiple plaintext values in each ciphertext.
    * The maximum number of slots depends on a security parameter called ring
    * dimension. In this instance, we don't specify the ring dimension directly,
    * but let the library choose it for us, based on the security level we
    * choose, the multiplicative depth we want to support, and the scaling factor
    * size.
    *
    * Please use method GetRingDimension() to find out the exact ring dimension
    * being used for these parameters. Give ring dimension N, the maximum batch
    * size is N/2, because of the way CKKS works.
    */
     // uint32_t batchSize = 8;

     /* A4) Desired security level based on FHE standards.
    * This parameter can take four values. Three of the possible values
    * correspond to 128-bit, 192-bit, and 256-bit security, and the fourth value
    * corresponds to "NotSet", which means that the user is responsible for
    * choosing security parameters. Naturally, "NotSet" should be used only in
    * non-production environments, or by experts who understand the security
    * implications of their choices.
    *
    * If a given security level is selected, the library will consult the current
    * security parameter tables defined by the FHE standards consortium
    * (https://homomorphicencryption.org/introduction/) to automatically
    * select the security parameters. Please see "TABLES of RECOMMENDED
    * PARAMETERS" in  the following reference for more details:
    * http://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf
    */
     CCParams<CryptoContextCKKSRNS> parameters;
    //  parameters.SetSecurityLevel(HEStd_128_classic);
    parameters.SetSecurityLevel(HEStd_NotSet);
    // Inputs
#ifdef CROSS_LIBRARY_REFERENCE
    uint32_t batchSize = 8;
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25};
#else
    uint32_t batchSize = 8192;
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25};
#endif
    parameters.SetBatchSize(batchSize);

  // Number of towers in CKKS (for a fresh ciphertext with maximum number of levels)
  //  = the multiplicative depth + 1 (for decryption in FIXED* and FLEXIBLEAUTO modes)
  //  or  = the multiplicative depth + 2 (for FLEXIBLEAUTOEXT mode). 
  // One tower is always needed for decryption. 
  // Another tower is used by FLEXIBLEAUTOEXT to achieve higher CKKS precision.
//  #define HE_CraterLake  // Needs cmake .. -DNATIVEINT=32
//  #define HE_BASALISC
//  #define HE_FAB
//  #define HE_Cheddar
//  #define HE_FIDESlib
//  #define HE_BASALISC_CROSS  // Needs cmake .. -DNATIVEINT=32
//  #define HE_FAB_CROSS  // Needs cmake .. -DNATIVEINT=32
//  #define HE_Cheddar_CROSS  // Needs cmake .. -DNATIVEINT=32
//  #define HE_FIDESlib_CROSS // Needs cmake .. -DNATIVEINT=32

#ifdef CROSS_LIBRARY_REFERENCE
    parameters.SetRingDim(16);              // Degree
    parameters.SetMultiplicativeDepth(3);   // Number of limbs - 2/1
    parameters.SetFirstModSize(28);
    parameters.SetScalingModSize(28);       // Log q
    parameters.SetNumLargeDigits(3);
    #elif defined(HE_CraterLake) // same as HE_TABVI_SETUP_6
    parameters.SetRingDim(65536);           // Degree
    parameters.SetMultiplicativeDepth(49);  // Number of limbs - 2/1
    parameters.SetFirstModSize(27);
    parameters.SetScalingModSize(27);       // Log q
    parameters.SetNumLargeDigits(3);
#elif defined(HE_BASALISC_CROSS)
    parameters.SetRingDim(65536);           // Degree
    parameters.SetMultiplicativeDepth(45);  // 32 originally, needs 31 levels more, ceil up to 33/3 = 11 -> 11x4=44 number of limbs -> 44+3=47. which gives 45 multiplication depth
    parameters.SetFirstModSize(27);
    parameters.SetScalingModSize(27);       // Log q=40, could use rational rescaling, which adds 4 30-bit primes per three levels. use 27 to calculate the numbers.
    parameters.SetNumLargeDigits(3);
#elif defined(HE_FAB_CROSS)
    parameters.SetRingDim(65536);           // Degree
    parameters.SetMultiplicativeDepth(62);  // 32 limbs, which gets doubled to 64 limbs, which gives 62 multiplications depth
    parameters.SetFirstModSize(27);
    parameters.SetScalingModSize(27);       // Log q=52, needs to use double rescaling. double number of limbs.
    parameters.SetNumLargeDigits(4);
#elif defined(HE_Cheddar_CROSS)
    parameters.SetRingDim(65536);           // Degree
    parameters.SetMultiplicativeDepth(46);  // 48 limbs, which gives 46 multiplications depth to ensure the same input ciphertext sizes.
    parameters.SetFirstModSize(27);
    parameters.SetScalingModSize(27);       // 25-30 primes, just use 28 to calculate the numbers.
    parameters.SetNumLargeDigits(12);
#elif defined(HE_FIDESlib_CROSS)
    parameters.SetRingDim(65536);           // Degree
    parameters.SetMultiplicativeDepth(58);  // Original limbs as 30, double number of limbs -> 60, which gives 58 multiplications depth.
    parameters.SetFirstModSize(27);
    parameters.SetScalingModSize(27);       // Original q is 59, needs to use 2 <31-bit q in the double rescaling -- double number of limbs
    parameters.SetNumLargeDigits(3);
#elif defined(HE_BASALISC)
    parameters.SetRingDim(65536);           // Degree
    parameters.SetMultiplicativeDepth(30);  // Number of limbs - 2/1
    parameters.SetFirstModSize(40);
    parameters.SetScalingModSize(40);       // Log q
    parameters.SetNumLargeDigits(3);
#elif defined(HE_FAB)
    parameters.SetRingDim(65536);           // Degree
    parameters.SetMultiplicativeDepth(30);  // Number of limbs - 2/1
    parameters.SetFirstModSize(52);
    parameters.SetScalingModSize(52);       // Log q
    parameters.SetNumLargeDigits(4);
#elif defined(HE_Cheddar)
    parameters.SetRingDim(65536);           // Degree
    parameters.SetMultiplicativeDepth(46);  // Number of limbs - 2/1
    parameters.SetFirstModSize(28);
    parameters.SetScalingModSize(28);       // Log q
    parameters.SetNumLargeDigits(12);
#elif defined(HE_FIDESlib)
    parameters.SetRingDim(65536);           // Degree
    parameters.SetMultiplicativeDepth(28);  // Number of limbs - 2/1
    parameters.SetFirstModSize(59);
    parameters.SetScalingModSize(59);       // Log q
    parameters.SetNumLargeDigits(3);
#elif defined(HE_TABVI_SETUP_OTHER)         // Not used in the paper
    parameters.SetRingDim(131072);          // Degree
    parameters.SetMultiplicativeDepth(34);  // Number of limbs - 2/1
    parameters.SetFirstModSize(59);
    parameters.SetScalingModSize(59);       // Log q
    parameters.SetNumLargeDigits(3);
#endif

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    // Enable the features that you wish to use
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

#ifdef PROFILE_ALL_DATA
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    std::cout << "Modulus bits: " << cc->GetModulus().GetLengthForBase(2) << std::endl;
    std::cout << "Modulus Limbs: " << cc->GetModulus().GetNumberOfLimbs() << std::endl;
    std::cout << "Modulus: " << cc->GetModulus() << std::endl;
    std::cout << "GetRootOfUnity: " << cc->GetRootOfUnity() << std::endl;
    std::cout << "GetEncodingParams: " << cc->GetEncodingParams() << std::endl;
    std::cout << "GetCyclotomicOrder: " << cc->GetCyclotomicOrder() << std::endl;
#endif

     // B. Step 2: Key Generation
     /* B1) Generate encryption keys.
    * These are used for encryption/decryption, as well as in generating
    * different kinds of keys.
    */
    auto keys = cc->KeyGen();


#ifdef PROFILE_ALL_MODULI
    // Print extended modulus (Q*P) from the crypto context
    try {
        const auto cryptoParamsRNS = std::dynamic_pointer_cast<CryptoParametersRNS>(cc->GetCryptoParameters());
        if (cryptoParamsRNS) {
            const auto paramsQP = cryptoParamsRNS->GetParamsQP();
            std::cout << "Extended modulus (QP) bits: " << paramsQP->GetModulus().GetMSB() << std::endl;
            std::cout << "Extended modulus (QP): " << paramsQP->GetModulus() << std::endl;

            const auto paramsQ = cryptoParamsRNS->GetElementParams();
            const auto paramsP = cryptoParamsRNS->GetParamsP();
            // Print individual moduli for non-extended part (Q)
            if (paramsQ) {
                const auto& qParamsVec = paramsQ->GetParams();
                std::cout << "Non-extended Q towers (" << qParamsVec.size() << ")" << std::endl;
                for (size_t i = 0; i < qParamsVec.size(); ++i) {
                    const auto& qi = qParamsVec[i]->GetModulus();
                    std::cout << "  Q[" << i << "] bits: " << qi.GetMSB() << ", modulus: " << qi << std::endl;
                }
            }

            // Print individual moduli for extended part (P)
            if (paramsP) {
                const auto& pParamsVec = paramsP->GetParams();
                std::cout << "Extended P towers (" << pParamsVec.size() << ")" << std::endl;
                for (size_t i = 0; i < pParamsVec.size(); ++i) {
                    const auto& pi = pParamsVec[i]->GetModulus();
                    std::cout << "  P[" << i << "] bits: " << pi.GetMSB() << ", modulus: " << pi << std::endl;
                }
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to retrieve extended modulus (QP): " << e.what() << std::endl;
    }
#endif

     /* B3) Generate the rotation keys
    * CKKS supports rotating the contents of a packed ciphertext, but to do so,
    * we need to create what we call a rotation key. This is done with the
    * following call, which takes as input a vector with indices that correspond
    * to the rotation offset we want to support. Negative indices correspond to
    * right shift and positive to left shift. Look at the output of this demo for
    * an illustration of this.
    *
    * Keep in mind that rotations work over the batch size or entire ring dimension (if the batch size is not specified).
    * This means that, if ring dimension is 8 and batch
    * size is not specified, then an input (1,2,3,4,0,0,0,0) rotated by 2 will become
    * (3,4,0,0,0,0,1,2) and not (3,4,1,2,0,0,0,0).
    * If ring dimension is 8 and batch
    * size is set to 4, then the rotation of (1,2,3,4) by 2 will become (3,4,1,2).
    * Also, as someone can observe
    * in the output of this demo, since CKKS is approximate, zeros are not exact
    * - they're just very small numbers.
    */
    std::cout << "!!!!!!!!!!!!!!!!!! EvalRotateKeyGen !!!!!!!!!!!!!!!!!!"<< std::endl;
    cc->EvalRotateKeyGen(keys.secretKey, {1});

#ifdef DEBUG_PRINT_KEYS
    try {
        auto& evalKeyMap = cc->GetEvalAutomorphismKeyMap(keys.secretKey->GetKeyTag());
        std::cout << "EvalAutomorphismKey map size: " << evalKeyMap.size() << std::endl;
        for (const auto& kv : evalKeyMap) {
            std::cout << "[AutoIndex " << kv.first << "]" << std::endl;
            const auto& ek   = kv.second;
            const auto& aVec = ek->GetAVector();
            const auto& bVec = ek->GetBVector();
            std::cout << "  AVector size: " << aVec.size() << std::endl;
            std::cout << "  BVector size: " << bVec.size() << std::endl;
            for (size_t i = 0; i < aVec.size(); ++i) {
                std::cout << "Element " << i << ": " << aVec[i] << std::endl;
            }
            std::cout << "  BVector size: " << bVec.size() << std::endl;
            for (size_t i = 0; i < bVec.size(); ++i) {
                std::cout << "Element " << i << ": " << bVec[i] << std::endl;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to retrieve/print EvalAutomorphismKey map: " << e.what() << std::endl;
    }
#endif

    // Step 3: Encoding and encryption of inputs
    // Encoding as plaintexts
    std::cout << "!!!!!!!!!!!!!!!!!! ptxt1 !!!!!!!!!!!!!!!!!!"<< std::endl;
    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    std::cout << "x1: " << x1 << "its encoded plaintext is: " << ptxt1 << std::endl;
    // std::cout << "ptxt1: " << ptxt1->GetCKKSPackedValue() << std::endl;

    // Encrypt the encoded vectors
    std::cout << "!!!!!!!!!!!!!!!!!! Encrypting ptxt1 !!!!!!!!!!!!!!!!!!"<< std::endl;
    std::cout << "ptxt1: " << ptxt1 << std::endl;
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    std::cout << "!!!!!!!!!!!!!!!!!! $$$$$ FRIST COPY POINT $$$$$ !!!!!!!!!!!!!!!!!!"<< std::endl;
    std::cout << "c1: " << c1 << std::endl;

    // Print paramsQl, paramsP, paramsQlP
#ifdef DEBUG_PRINT_KEYS
    try {
        const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersRNS>(cc->GetCryptoParameters());
        const auto paramsQl  = c1->GetElements()[0].GetParams();
        const auto paramsP   = cryptoParams->GetParamsP();
        const auto paramsQlP = c1->GetElements()[0].GetExtendedCRTBasis(paramsP);

        std::cout << "!!!!!!!!!!!!!!!!!! Params Ql/P/QlP !!!!!!!!!!!!!!!!!!" << std::endl;
        std::cout << "sizeQl=" << paramsQl->GetParams().size()
                  << ", sizeP=" << paramsP->GetParams().size()
                  << ", sizeQlP=" << paramsQlP->GetParams().size() << std::endl;
        std::cout << "RingDim Ql=" << paramsQl->GetRingDimension()
                  << ", P=" << paramsP->GetRingDimension()
                  << ", QlP=" << paramsQlP->GetRingDimension() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to retrieve/print paramsQl/paramsP/paramsQlP: " << e.what() << std::endl;
    }
#endif

    std::cout << "!!!!!!!!!!!!!!!!!! cRot1 = cc->EvalRotate(c1, 1); !!!!!!!!!!!!!!!!!!"<< std::endl;
    auto cRot1 = cc->EvalRotate(c1, 1);

    // Step 5: End-to-end evaluation + decryption
    // Homomorphic multiplication
    Plaintext result;
    std::cout << std::endl << "Results of homomorphic computations: " << std::endl;
    cc->Decrypt(keys.secretKey, cRot1, &result);
    result->SetLength(batchSize);
    std::cout << std::endl << "In rotations, very small outputs (~10^-10 here) correspond to 0's:" << std::endl;
    std::cout << "x1 rotate by 1 = " << result << std::endl;

    return 0;
 }
