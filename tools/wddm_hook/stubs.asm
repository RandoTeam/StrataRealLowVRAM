.code
EXTERN g_real_cublas_funcs: QWORD

PUBLIC cublasAlloc
cublasAlloc PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 0]
    jmp rax
cublasAlloc ENDP

PUBLIC cublasAsumEx
cublasAsumEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 8]
    jmp rax
cublasAsumEx ENDP

PUBLIC cublasAsumEx_64
cublasAsumEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 16]
    jmp rax
cublasAsumEx_64 ENDP

PUBLIC cublasAxpyEx
cublasAxpyEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 24]
    jmp rax
cublasAxpyEx ENDP

PUBLIC cublasAxpyEx_64
cublasAxpyEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 32]
    jmp rax
cublasAxpyEx_64 ENDP

PUBLIC cublasCaxpy
cublasCaxpy PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 40]
    jmp rax
cublasCaxpy ENDP

PUBLIC cublasCaxpy_v2
cublasCaxpy_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 48]
    jmp rax
cublasCaxpy_v2 ENDP

PUBLIC cublasCaxpy_v2_64
cublasCaxpy_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 56]
    jmp rax
cublasCaxpy_v2_64 ENDP

PUBLIC cublasCbdmm
cublasCbdmm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 64]
    jmp rax
cublasCbdmm ENDP

PUBLIC cublasCcopy
cublasCcopy PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 72]
    jmp rax
cublasCcopy ENDP

PUBLIC cublasCcopy_v2
cublasCcopy_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 80]
    jmp rax
cublasCcopy_v2 ENDP

PUBLIC cublasCcopy_v2_64
cublasCcopy_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 88]
    jmp rax
cublasCcopy_v2_64 ENDP

PUBLIC cublasCdgmm
cublasCdgmm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 96]
    jmp rax
cublasCdgmm ENDP

PUBLIC cublasCdgmm_64
cublasCdgmm_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 104]
    jmp rax
cublasCdgmm_64 ENDP

PUBLIC cublasCdotc
cublasCdotc PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 112]
    jmp rax
cublasCdotc ENDP

PUBLIC cublasCdotc_v2
cublasCdotc_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 120]
    jmp rax
cublasCdotc_v2 ENDP

PUBLIC cublasCdotc_v2_64
cublasCdotc_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 128]
    jmp rax
cublasCdotc_v2_64 ENDP

PUBLIC cublasCdotu
cublasCdotu PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 136]
    jmp rax
cublasCdotu ENDP

PUBLIC cublasCdotu_v2
cublasCdotu_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 144]
    jmp rax
cublasCdotu_v2 ENDP

PUBLIC cublasCdotu_v2_64
cublasCdotu_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 152]
    jmp rax
cublasCdotu_v2_64 ENDP

PUBLIC cublasCgbmv
cublasCgbmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 160]
    jmp rax
cublasCgbmv ENDP

PUBLIC cublasCgbmv_v2
cublasCgbmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 168]
    jmp rax
cublasCgbmv_v2 ENDP

PUBLIC cublasCgbmv_v2_64
cublasCgbmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 176]
    jmp rax
cublasCgbmv_v2_64 ENDP

PUBLIC cublasCgeam
cublasCgeam PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 184]
    jmp rax
cublasCgeam ENDP

PUBLIC cublasCgeam_64
cublasCgeam_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 192]
    jmp rax
cublasCgeam_64 ENDP

PUBLIC cublasCgelsBatched
cublasCgelsBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 200]
    jmp rax
cublasCgelsBatched ENDP

PUBLIC cublasCgemm
cublasCgemm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 208]
    jmp rax
cublasCgemm ENDP

PUBLIC cublasCgemm3m
cublasCgemm3m PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 216]
    jmp rax
cublasCgemm3m ENDP

PUBLIC cublasCgemm3mBatched
cublasCgemm3mBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 224]
    jmp rax
cublasCgemm3mBatched ENDP

PUBLIC cublasCgemm3mBatched_64
cublasCgemm3mBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 232]
    jmp rax
cublasCgemm3mBatched_64 ENDP

PUBLIC cublasCgemm3mEx
cublasCgemm3mEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 240]
    jmp rax
cublasCgemm3mEx ENDP

PUBLIC cublasCgemm3mEx_64
cublasCgemm3mEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 248]
    jmp rax
cublasCgemm3mEx_64 ENDP

PUBLIC cublasCgemm3mStridedBatched
cublasCgemm3mStridedBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 256]
    jmp rax
cublasCgemm3mStridedBatched ENDP

PUBLIC cublasCgemm3mStridedBatched_64
cublasCgemm3mStridedBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 264]
    jmp rax
cublasCgemm3mStridedBatched_64 ENDP

PUBLIC cublasCgemm3m_64
cublasCgemm3m_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 272]
    jmp rax
cublasCgemm3m_64 ENDP

PUBLIC cublasCgemmBatched
cublasCgemmBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 280]
    jmp rax
cublasCgemmBatched ENDP

PUBLIC cublasCgemmBatched_64
cublasCgemmBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 288]
    jmp rax
cublasCgemmBatched_64 ENDP

PUBLIC cublasCgemmEx
cublasCgemmEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 296]
    jmp rax
cublasCgemmEx ENDP

PUBLIC cublasCgemmEx_64
cublasCgemmEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 304]
    jmp rax
cublasCgemmEx_64 ENDP

PUBLIC cublasCgemmStridedBatched
cublasCgemmStridedBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 312]
    jmp rax
cublasCgemmStridedBatched ENDP

PUBLIC cublasCgemmStridedBatched_64
cublasCgemmStridedBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 320]
    jmp rax
cublasCgemmStridedBatched_64 ENDP

PUBLIC cublasCgemm_v2
cublasCgemm_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 328]
    jmp rax
cublasCgemm_v2 ENDP

PUBLIC cublasCgemm_v2_64
cublasCgemm_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 336]
    jmp rax
cublasCgemm_v2_64 ENDP

PUBLIC cublasCgemv
cublasCgemv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 344]
    jmp rax
cublasCgemv ENDP

PUBLIC cublasCgemvBatched
cublasCgemvBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 352]
    jmp rax
cublasCgemvBatched ENDP

PUBLIC cublasCgemvBatched_64
cublasCgemvBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 360]
    jmp rax
cublasCgemvBatched_64 ENDP

PUBLIC cublasCgemvStridedBatched
cublasCgemvStridedBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 368]
    jmp rax
cublasCgemvStridedBatched ENDP

PUBLIC cublasCgemvStridedBatched_64
cublasCgemvStridedBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 376]
    jmp rax
cublasCgemvStridedBatched_64 ENDP

PUBLIC cublasCgemv_v2
cublasCgemv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 384]
    jmp rax
cublasCgemv_v2 ENDP

PUBLIC cublasCgemv_v2_64
cublasCgemv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 392]
    jmp rax
cublasCgemv_v2_64 ENDP

PUBLIC cublasCgeqrfBatched
cublasCgeqrfBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 400]
    jmp rax
cublasCgeqrfBatched ENDP

PUBLIC cublasCgerc
cublasCgerc PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 408]
    jmp rax
cublasCgerc ENDP

PUBLIC cublasCgerc_v2
cublasCgerc_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 416]
    jmp rax
cublasCgerc_v2 ENDP

PUBLIC cublasCgerc_v2_64
cublasCgerc_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 424]
    jmp rax
cublasCgerc_v2_64 ENDP

PUBLIC cublasCgeru
cublasCgeru PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 432]
    jmp rax
cublasCgeru ENDP

PUBLIC cublasCgeru_v2
cublasCgeru_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 440]
    jmp rax
cublasCgeru_v2 ENDP

PUBLIC cublasCgeru_v2_64
cublasCgeru_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 448]
    jmp rax
cublasCgeru_v2_64 ENDP

PUBLIC cublasCgetrfBatched
cublasCgetrfBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 456]
    jmp rax
cublasCgetrfBatched ENDP

PUBLIC cublasCgetriBatched
cublasCgetriBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 464]
    jmp rax
cublasCgetriBatched ENDP

PUBLIC cublasCgetrsBatched
cublasCgetrsBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 472]
    jmp rax
cublasCgetrsBatched ENDP

PUBLIC cublasChbmv
cublasChbmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 480]
    jmp rax
cublasChbmv ENDP

PUBLIC cublasChbmv_v2
cublasChbmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 488]
    jmp rax
cublasChbmv_v2 ENDP

PUBLIC cublasChbmv_v2_64
cublasChbmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 496]
    jmp rax
cublasChbmv_v2_64 ENDP

PUBLIC cublasChemm
cublasChemm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 504]
    jmp rax
cublasChemm ENDP

PUBLIC cublasChemm_v2
cublasChemm_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 512]
    jmp rax
cublasChemm_v2 ENDP

PUBLIC cublasChemm_v2_64
cublasChemm_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 520]
    jmp rax
cublasChemm_v2_64 ENDP

PUBLIC cublasChemv
cublasChemv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 528]
    jmp rax
cublasChemv ENDP

PUBLIC cublasChemv_v2
cublasChemv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 536]
    jmp rax
cublasChemv_v2 ENDP

PUBLIC cublasChemv_v2_64
cublasChemv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 544]
    jmp rax
cublasChemv_v2_64 ENDP

PUBLIC cublasCher
cublasCher PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 552]
    jmp rax
cublasCher ENDP

PUBLIC cublasCher2
cublasCher2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 560]
    jmp rax
cublasCher2 ENDP

PUBLIC cublasCher2_v2
cublasCher2_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 568]
    jmp rax
cublasCher2_v2 ENDP

PUBLIC cublasCher2_v2_64
cublasCher2_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 576]
    jmp rax
cublasCher2_v2_64 ENDP

PUBLIC cublasCher2k
cublasCher2k PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 584]
    jmp rax
cublasCher2k ENDP

PUBLIC cublasCher2k_v2
cublasCher2k_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 592]
    jmp rax
cublasCher2k_v2 ENDP

PUBLIC cublasCher2k_v2_64
cublasCher2k_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 600]
    jmp rax
cublasCher2k_v2_64 ENDP

PUBLIC cublasCher_v2
cublasCher_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 608]
    jmp rax
cublasCher_v2 ENDP

PUBLIC cublasCher_v2_64
cublasCher_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 616]
    jmp rax
cublasCher_v2_64 ENDP

PUBLIC cublasCherk
cublasCherk PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 624]
    jmp rax
cublasCherk ENDP

PUBLIC cublasCherk3mEx
cublasCherk3mEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 632]
    jmp rax
cublasCherk3mEx ENDP

PUBLIC cublasCherk3mEx_64
cublasCherk3mEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 640]
    jmp rax
cublasCherk3mEx_64 ENDP

PUBLIC cublasCherkEx
cublasCherkEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 648]
    jmp rax
cublasCherkEx ENDP

PUBLIC cublasCherkEx_64
cublasCherkEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 656]
    jmp rax
cublasCherkEx_64 ENDP

PUBLIC cublasCherk_v2
cublasCherk_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 664]
    jmp rax
cublasCherk_v2 ENDP

PUBLIC cublasCherk_v2_64
cublasCherk_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 672]
    jmp rax
cublasCherk_v2_64 ENDP

PUBLIC cublasCherkx
cublasCherkx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 680]
    jmp rax
cublasCherkx ENDP

PUBLIC cublasCherkx_64
cublasCherkx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 688]
    jmp rax
cublasCherkx_64 ENDP

PUBLIC cublasChpmv
cublasChpmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 696]
    jmp rax
cublasChpmv ENDP

PUBLIC cublasChpmv_v2
cublasChpmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 704]
    jmp rax
cublasChpmv_v2 ENDP

PUBLIC cublasChpmv_v2_64
cublasChpmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 712]
    jmp rax
cublasChpmv_v2_64 ENDP

PUBLIC cublasChpr
cublasChpr PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 720]
    jmp rax
cublasChpr ENDP

PUBLIC cublasChpr2
cublasChpr2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 728]
    jmp rax
cublasChpr2 ENDP

PUBLIC cublasChpr2_v2
cublasChpr2_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 736]
    jmp rax
cublasChpr2_v2 ENDP

PUBLIC cublasChpr2_v2_64
cublasChpr2_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 744]
    jmp rax
cublasChpr2_v2_64 ENDP

PUBLIC cublasChpr_v2
cublasChpr_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 752]
    jmp rax
cublasChpr_v2 ENDP

PUBLIC cublasChpr_v2_64
cublasChpr_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 760]
    jmp rax
cublasChpr_v2_64 ENDP

PUBLIC cublasCmatinvBatched
cublasCmatinvBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 768]
    jmp rax
cublasCmatinvBatched ENDP

PUBLIC cublasCopyEx
cublasCopyEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 776]
    jmp rax
cublasCopyEx ENDP

PUBLIC cublasCopyEx_64
cublasCopyEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 784]
    jmp rax
cublasCopyEx_64 ENDP

PUBLIC cublasCreate_v2
cublasCreate_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 792]
    jmp rax
cublasCreate_v2 ENDP

PUBLIC cublasCrot
cublasCrot PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 800]
    jmp rax
cublasCrot ENDP

PUBLIC cublasCrot_v2
cublasCrot_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 808]
    jmp rax
cublasCrot_v2 ENDP

PUBLIC cublasCrot_v2_64
cublasCrot_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 816]
    jmp rax
cublasCrot_v2_64 ENDP

PUBLIC cublasCrotg
cublasCrotg PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 824]
    jmp rax
cublasCrotg ENDP

PUBLIC cublasCrotg_v2
cublasCrotg_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 832]
    jmp rax
cublasCrotg_v2 ENDP

PUBLIC cublasCscal
cublasCscal PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 840]
    jmp rax
cublasCscal ENDP

PUBLIC cublasCscal_v2
cublasCscal_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 848]
    jmp rax
cublasCscal_v2 ENDP

PUBLIC cublasCscal_v2_64
cublasCscal_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 856]
    jmp rax
cublasCscal_v2_64 ENDP

PUBLIC cublasCsrot
cublasCsrot PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 864]
    jmp rax
cublasCsrot ENDP

PUBLIC cublasCsrot_v2
cublasCsrot_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 872]
    jmp rax
cublasCsrot_v2 ENDP

PUBLIC cublasCsrot_v2_64
cublasCsrot_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 880]
    jmp rax
cublasCsrot_v2_64 ENDP

PUBLIC cublasCsscal
cublasCsscal PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 888]
    jmp rax
cublasCsscal ENDP

PUBLIC cublasCsscal_v2
cublasCsscal_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 896]
    jmp rax
cublasCsscal_v2 ENDP

PUBLIC cublasCsscal_v2_64
cublasCsscal_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 904]
    jmp rax
cublasCsscal_v2_64 ENDP

PUBLIC cublasCswap
cublasCswap PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 912]
    jmp rax
cublasCswap ENDP

PUBLIC cublasCswap_v2
cublasCswap_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 920]
    jmp rax
cublasCswap_v2 ENDP

PUBLIC cublasCswap_v2_64
cublasCswap_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 928]
    jmp rax
cublasCswap_v2_64 ENDP

PUBLIC cublasCsymm
cublasCsymm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 936]
    jmp rax
cublasCsymm ENDP

PUBLIC cublasCsymm_v2
cublasCsymm_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 944]
    jmp rax
cublasCsymm_v2 ENDP

PUBLIC cublasCsymm_v2_64
cublasCsymm_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 952]
    jmp rax
cublasCsymm_v2_64 ENDP

PUBLIC cublasCsymv_v2
cublasCsymv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 960]
    jmp rax
cublasCsymv_v2 ENDP

PUBLIC cublasCsymv_v2_64
cublasCsymv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 968]
    jmp rax
cublasCsymv_v2_64 ENDP

PUBLIC cublasCsyr2_v2
cublasCsyr2_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 976]
    jmp rax
cublasCsyr2_v2 ENDP

PUBLIC cublasCsyr2_v2_64
cublasCsyr2_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 984]
    jmp rax
cublasCsyr2_v2_64 ENDP

PUBLIC cublasCsyr2k
cublasCsyr2k PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 992]
    jmp rax
cublasCsyr2k ENDP

PUBLIC cublasCsyr2k_v2
cublasCsyr2k_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1000]
    jmp rax
cublasCsyr2k_v2 ENDP

PUBLIC cublasCsyr2k_v2_64
cublasCsyr2k_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1008]
    jmp rax
cublasCsyr2k_v2_64 ENDP

PUBLIC cublasCsyr_v2
cublasCsyr_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1016]
    jmp rax
cublasCsyr_v2 ENDP

PUBLIC cublasCsyr_v2_64
cublasCsyr_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1024]
    jmp rax
cublasCsyr_v2_64 ENDP

PUBLIC cublasCsyrk
cublasCsyrk PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1032]
    jmp rax
cublasCsyrk ENDP

PUBLIC cublasCsyrk3mEx
cublasCsyrk3mEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1040]
    jmp rax
cublasCsyrk3mEx ENDP

PUBLIC cublasCsyrk3mEx_64
cublasCsyrk3mEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1048]
    jmp rax
cublasCsyrk3mEx_64 ENDP

PUBLIC cublasCsyrkEx
cublasCsyrkEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1056]
    jmp rax
cublasCsyrkEx ENDP

PUBLIC cublasCsyrkEx_64
cublasCsyrkEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1064]
    jmp rax
cublasCsyrkEx_64 ENDP

PUBLIC cublasCsyrk_v2
cublasCsyrk_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1072]
    jmp rax
cublasCsyrk_v2 ENDP

PUBLIC cublasCsyrk_v2_64
cublasCsyrk_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1080]
    jmp rax
cublasCsyrk_v2_64 ENDP

PUBLIC cublasCsyrkx
cublasCsyrkx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1088]
    jmp rax
cublasCsyrkx ENDP

PUBLIC cublasCsyrkx_64
cublasCsyrkx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1096]
    jmp rax
cublasCsyrkx_64 ENDP

PUBLIC cublasCtbmv
cublasCtbmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1104]
    jmp rax
cublasCtbmv ENDP

PUBLIC cublasCtbmv_v2
cublasCtbmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1112]
    jmp rax
cublasCtbmv_v2 ENDP

PUBLIC cublasCtbmv_v2_64
cublasCtbmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1120]
    jmp rax
cublasCtbmv_v2_64 ENDP

PUBLIC cublasCtbsv
cublasCtbsv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1128]
    jmp rax
cublasCtbsv ENDP

PUBLIC cublasCtbsv_v2
cublasCtbsv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1136]
    jmp rax
cublasCtbsv_v2 ENDP

PUBLIC cublasCtbsv_v2_64
cublasCtbsv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1144]
    jmp rax
cublasCtbsv_v2_64 ENDP

PUBLIC cublasCtpmv
cublasCtpmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1152]
    jmp rax
cublasCtpmv ENDP

PUBLIC cublasCtpmv_v2
cublasCtpmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1160]
    jmp rax
cublasCtpmv_v2 ENDP

PUBLIC cublasCtpmv_v2_64
cublasCtpmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1168]
    jmp rax
cublasCtpmv_v2_64 ENDP

PUBLIC cublasCtpsv
cublasCtpsv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1176]
    jmp rax
cublasCtpsv ENDP

PUBLIC cublasCtpsv_v2
cublasCtpsv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1184]
    jmp rax
cublasCtpsv_v2 ENDP

PUBLIC cublasCtpsv_v2_64
cublasCtpsv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1192]
    jmp rax
cublasCtpsv_v2_64 ENDP

PUBLIC cublasCtpttr
cublasCtpttr PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1200]
    jmp rax
cublasCtpttr ENDP

PUBLIC cublasCtrmm
cublasCtrmm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1208]
    jmp rax
cublasCtrmm ENDP

PUBLIC cublasCtrmm_v2
cublasCtrmm_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1216]
    jmp rax
cublasCtrmm_v2 ENDP

PUBLIC cublasCtrmm_v2_64
cublasCtrmm_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1224]
    jmp rax
cublasCtrmm_v2_64 ENDP

PUBLIC cublasCtrmv
cublasCtrmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1232]
    jmp rax
cublasCtrmv ENDP

PUBLIC cublasCtrmv_v2
cublasCtrmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1240]
    jmp rax
cublasCtrmv_v2 ENDP

PUBLIC cublasCtrmv_v2_64
cublasCtrmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1248]
    jmp rax
cublasCtrmv_v2_64 ENDP

PUBLIC cublasCtrsm
cublasCtrsm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1256]
    jmp rax
cublasCtrsm ENDP

PUBLIC cublasCtrsmBatched
cublasCtrsmBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1264]
    jmp rax
cublasCtrsmBatched ENDP

PUBLIC cublasCtrsmBatched_64
cublasCtrsmBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1272]
    jmp rax
cublasCtrsmBatched_64 ENDP

PUBLIC cublasCtrsm_v2
cublasCtrsm_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1280]
    jmp rax
cublasCtrsm_v2 ENDP

PUBLIC cublasCtrsm_v2_64
cublasCtrsm_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1288]
    jmp rax
cublasCtrsm_v2_64 ENDP

PUBLIC cublasCtrsv
cublasCtrsv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1296]
    jmp rax
cublasCtrsv ENDP

PUBLIC cublasCtrsv_v2
cublasCtrsv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1304]
    jmp rax
cublasCtrsv_v2 ENDP

PUBLIC cublasCtrsv_v2_64
cublasCtrsv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1312]
    jmp rax
cublasCtrsv_v2_64 ENDP

PUBLIC cublasCtrttp
cublasCtrttp PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1320]
    jmp rax
cublasCtrttp ENDP

PUBLIC cublasDasum
cublasDasum PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1328]
    jmp rax
cublasDasum ENDP

PUBLIC cublasDasum_v2
cublasDasum_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1336]
    jmp rax
cublasDasum_v2 ENDP

PUBLIC cublasDasum_v2_64
cublasDasum_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1344]
    jmp rax
cublasDasum_v2_64 ENDP

PUBLIC cublasDaxpy
cublasDaxpy PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1352]
    jmp rax
cublasDaxpy ENDP

PUBLIC cublasDaxpy_v2
cublasDaxpy_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1360]
    jmp rax
cublasDaxpy_v2 ENDP

PUBLIC cublasDaxpy_v2_64
cublasDaxpy_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1368]
    jmp rax
cublasDaxpy_v2_64 ENDP

PUBLIC cublasDbdmm
cublasDbdmm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1376]
    jmp rax
cublasDbdmm ENDP

PUBLIC cublasDcopy
cublasDcopy PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1384]
    jmp rax
cublasDcopy ENDP

PUBLIC cublasDcopy_v2
cublasDcopy_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1392]
    jmp rax
cublasDcopy_v2 ENDP

PUBLIC cublasDcopy_v2_64
cublasDcopy_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1400]
    jmp rax
cublasDcopy_v2_64 ENDP

PUBLIC cublasDdgmm
cublasDdgmm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1408]
    jmp rax
cublasDdgmm ENDP

PUBLIC cublasDdgmm_64
cublasDdgmm_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1416]
    jmp rax
cublasDdgmm_64 ENDP

PUBLIC cublasDdot
cublasDdot PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1424]
    jmp rax
cublasDdot ENDP

PUBLIC cublasDdot_v2
cublasDdot_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1432]
    jmp rax
cublasDdot_v2 ENDP

PUBLIC cublasDdot_v2_64
cublasDdot_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1440]
    jmp rax
cublasDdot_v2_64 ENDP

PUBLIC cublasDestroy_v2
cublasDestroy_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1448]
    jmp rax
cublasDestroy_v2 ENDP

PUBLIC cublasDgbmv
cublasDgbmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1456]
    jmp rax
cublasDgbmv ENDP

PUBLIC cublasDgbmv_v2
cublasDgbmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1464]
    jmp rax
cublasDgbmv_v2 ENDP

PUBLIC cublasDgbmv_v2_64
cublasDgbmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1472]
    jmp rax
cublasDgbmv_v2_64 ENDP

PUBLIC cublasDgeam
cublasDgeam PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1480]
    jmp rax
cublasDgeam ENDP

PUBLIC cublasDgeam_64
cublasDgeam_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1488]
    jmp rax
cublasDgeam_64 ENDP

PUBLIC cublasDgelsBatched
cublasDgelsBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1496]
    jmp rax
cublasDgelsBatched ENDP

PUBLIC cublasDgemm
cublasDgemm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1504]
    jmp rax
cublasDgemm ENDP

PUBLIC cublasDgemmBatched
cublasDgemmBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1512]
    jmp rax
cublasDgemmBatched ENDP

PUBLIC cublasDgemmBatched_64
cublasDgemmBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1520]
    jmp rax
cublasDgemmBatched_64 ENDP

PUBLIC cublasDgemmGroupedBatched
cublasDgemmGroupedBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1528]
    jmp rax
cublasDgemmGroupedBatched ENDP

PUBLIC cublasDgemmGroupedBatched_64
cublasDgemmGroupedBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1536]
    jmp rax
cublasDgemmGroupedBatched_64 ENDP

PUBLIC cublasDgemmStridedBatched
cublasDgemmStridedBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1544]
    jmp rax
cublasDgemmStridedBatched ENDP

PUBLIC cublasDgemmStridedBatched_64
cublasDgemmStridedBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1552]
    jmp rax
cublasDgemmStridedBatched_64 ENDP

PUBLIC cublasDgemm_v2
cublasDgemm_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1560]
    jmp rax
cublasDgemm_v2 ENDP

PUBLIC cublasDgemm_v2_64
cublasDgemm_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1568]
    jmp rax
cublasDgemm_v2_64 ENDP

PUBLIC cublasDgemv
cublasDgemv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1576]
    jmp rax
cublasDgemv ENDP

PUBLIC cublasDgemvBatched
cublasDgemvBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1584]
    jmp rax
cublasDgemvBatched ENDP

PUBLIC cublasDgemvBatched_64
cublasDgemvBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1592]
    jmp rax
cublasDgemvBatched_64 ENDP

PUBLIC cublasDgemvStridedBatched
cublasDgemvStridedBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1600]
    jmp rax
cublasDgemvStridedBatched ENDP

PUBLIC cublasDgemvStridedBatched_64
cublasDgemvStridedBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1608]
    jmp rax
cublasDgemvStridedBatched_64 ENDP

PUBLIC cublasDgemv_v2
cublasDgemv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1616]
    jmp rax
cublasDgemv_v2 ENDP

PUBLIC cublasDgemv_v2_64
cublasDgemv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1624]
    jmp rax
cublasDgemv_v2_64 ENDP

PUBLIC cublasDgeqrfBatched
cublasDgeqrfBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1632]
    jmp rax
cublasDgeqrfBatched ENDP

PUBLIC cublasDger
cublasDger PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1640]
    jmp rax
cublasDger ENDP

PUBLIC cublasDger_v2
cublasDger_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1648]
    jmp rax
cublasDger_v2 ENDP

PUBLIC cublasDger_v2_64
cublasDger_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1656]
    jmp rax
cublasDger_v2_64 ENDP

PUBLIC cublasDgetrfBatched
cublasDgetrfBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1664]
    jmp rax
cublasDgetrfBatched ENDP

PUBLIC cublasDgetriBatched
cublasDgetriBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1672]
    jmp rax
cublasDgetriBatched ENDP

PUBLIC cublasDgetrsBatched
cublasDgetrsBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1680]
    jmp rax
cublasDgetrsBatched ENDP

PUBLIC cublasDmatinvBatched
cublasDmatinvBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1688]
    jmp rax
cublasDmatinvBatched ENDP

PUBLIC cublasDnrm2
cublasDnrm2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1696]
    jmp rax
cublasDnrm2 ENDP

PUBLIC cublasDnrm2_v2
cublasDnrm2_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1704]
    jmp rax
cublasDnrm2_v2 ENDP

PUBLIC cublasDnrm2_v2_64
cublasDnrm2_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1712]
    jmp rax
cublasDnrm2_v2_64 ENDP

PUBLIC cublasDotEx
cublasDotEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1720]
    jmp rax
cublasDotEx ENDP

PUBLIC cublasDotEx_64
cublasDotEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1728]
    jmp rax
cublasDotEx_64 ENDP

PUBLIC cublasDotcEx
cublasDotcEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1736]
    jmp rax
cublasDotcEx ENDP

PUBLIC cublasDotcEx_64
cublasDotcEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1744]
    jmp rax
cublasDotcEx_64 ENDP

PUBLIC cublasDrot
cublasDrot PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1752]
    jmp rax
cublasDrot ENDP

PUBLIC cublasDrot_v2
cublasDrot_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1760]
    jmp rax
cublasDrot_v2 ENDP

PUBLIC cublasDrot_v2_64
cublasDrot_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1768]
    jmp rax
cublasDrot_v2_64 ENDP

PUBLIC cublasDrotg
cublasDrotg PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1776]
    jmp rax
cublasDrotg ENDP

PUBLIC cublasDrotg_v2
cublasDrotg_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1784]
    jmp rax
cublasDrotg_v2 ENDP

PUBLIC cublasDrotm
cublasDrotm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1792]
    jmp rax
cublasDrotm ENDP

PUBLIC cublasDrotm_v2
cublasDrotm_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1800]
    jmp rax
cublasDrotm_v2 ENDP

PUBLIC cublasDrotm_v2_64
cublasDrotm_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1808]
    jmp rax
cublasDrotm_v2_64 ENDP

PUBLIC cublasDrotmg
cublasDrotmg PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1816]
    jmp rax
cublasDrotmg ENDP

PUBLIC cublasDrotmg_v2
cublasDrotmg_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1824]
    jmp rax
cublasDrotmg_v2 ENDP

PUBLIC cublasDsbmv
cublasDsbmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1832]
    jmp rax
cublasDsbmv ENDP

PUBLIC cublasDsbmv_v2
cublasDsbmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1840]
    jmp rax
cublasDsbmv_v2 ENDP

PUBLIC cublasDsbmv_v2_64
cublasDsbmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1848]
    jmp rax
cublasDsbmv_v2_64 ENDP

PUBLIC cublasDscal
cublasDscal PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1856]
    jmp rax
cublasDscal ENDP

PUBLIC cublasDscal_v2
cublasDscal_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1864]
    jmp rax
cublasDscal_v2 ENDP

PUBLIC cublasDscal_v2_64
cublasDscal_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1872]
    jmp rax
cublasDscal_v2_64 ENDP

PUBLIC cublasDspmv
cublasDspmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1880]
    jmp rax
cublasDspmv ENDP

PUBLIC cublasDspmv_v2
cublasDspmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1888]
    jmp rax
cublasDspmv_v2 ENDP

PUBLIC cublasDspmv_v2_64
cublasDspmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1896]
    jmp rax
cublasDspmv_v2_64 ENDP

PUBLIC cublasDspr
cublasDspr PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1904]
    jmp rax
cublasDspr ENDP

PUBLIC cublasDspr2
cublasDspr2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1912]
    jmp rax
cublasDspr2 ENDP

PUBLIC cublasDspr2_v2
cublasDspr2_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1920]
    jmp rax
cublasDspr2_v2 ENDP

PUBLIC cublasDspr2_v2_64
cublasDspr2_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1928]
    jmp rax
cublasDspr2_v2_64 ENDP

PUBLIC cublasDspr_v2
cublasDspr_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1936]
    jmp rax
cublasDspr_v2 ENDP

PUBLIC cublasDspr_v2_64
cublasDspr_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1944]
    jmp rax
cublasDspr_v2_64 ENDP

PUBLIC cublasDswap
cublasDswap PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1952]
    jmp rax
cublasDswap ENDP

PUBLIC cublasDswap_v2
cublasDswap_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1960]
    jmp rax
cublasDswap_v2 ENDP

PUBLIC cublasDswap_v2_64
cublasDswap_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1968]
    jmp rax
cublasDswap_v2_64 ENDP

PUBLIC cublasDsymm
cublasDsymm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1976]
    jmp rax
cublasDsymm ENDP

PUBLIC cublasDsymm_v2
cublasDsymm_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1984]
    jmp rax
cublasDsymm_v2 ENDP

PUBLIC cublasDsymm_v2_64
cublasDsymm_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 1992]
    jmp rax
cublasDsymm_v2_64 ENDP

PUBLIC cublasDsymv
cublasDsymv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2000]
    jmp rax
cublasDsymv ENDP

PUBLIC cublasDsymv_v2
cublasDsymv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2008]
    jmp rax
cublasDsymv_v2 ENDP

PUBLIC cublasDsymv_v2_64
cublasDsymv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2016]
    jmp rax
cublasDsymv_v2_64 ENDP

PUBLIC cublasDsyr
cublasDsyr PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2024]
    jmp rax
cublasDsyr ENDP

PUBLIC cublasDsyr2
cublasDsyr2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2032]
    jmp rax
cublasDsyr2 ENDP

PUBLIC cublasDsyr2_v2
cublasDsyr2_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2040]
    jmp rax
cublasDsyr2_v2 ENDP

PUBLIC cublasDsyr2_v2_64
cublasDsyr2_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2048]
    jmp rax
cublasDsyr2_v2_64 ENDP

PUBLIC cublasDsyr2k
cublasDsyr2k PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2056]
    jmp rax
cublasDsyr2k ENDP

PUBLIC cublasDsyr2k_v2
cublasDsyr2k_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2064]
    jmp rax
cublasDsyr2k_v2 ENDP

PUBLIC cublasDsyr2k_v2_64
cublasDsyr2k_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2072]
    jmp rax
cublasDsyr2k_v2_64 ENDP

PUBLIC cublasDsyr_v2
cublasDsyr_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2080]
    jmp rax
cublasDsyr_v2 ENDP

PUBLIC cublasDsyr_v2_64
cublasDsyr_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2088]
    jmp rax
cublasDsyr_v2_64 ENDP

PUBLIC cublasDsyrk
cublasDsyrk PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2096]
    jmp rax
cublasDsyrk ENDP

PUBLIC cublasDsyrk_v2
cublasDsyrk_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2104]
    jmp rax
cublasDsyrk_v2 ENDP

PUBLIC cublasDsyrk_v2_64
cublasDsyrk_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2112]
    jmp rax
cublasDsyrk_v2_64 ENDP

PUBLIC cublasDsyrkx
cublasDsyrkx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2120]
    jmp rax
cublasDsyrkx ENDP

PUBLIC cublasDsyrkx_64
cublasDsyrkx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2128]
    jmp rax
cublasDsyrkx_64 ENDP

PUBLIC cublasDtbmv
cublasDtbmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2136]
    jmp rax
cublasDtbmv ENDP

PUBLIC cublasDtbmv_v2
cublasDtbmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2144]
    jmp rax
cublasDtbmv_v2 ENDP

PUBLIC cublasDtbmv_v2_64
cublasDtbmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2152]
    jmp rax
cublasDtbmv_v2_64 ENDP

PUBLIC cublasDtbsv
cublasDtbsv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2160]
    jmp rax
cublasDtbsv ENDP

PUBLIC cublasDtbsv_v2
cublasDtbsv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2168]
    jmp rax
cublasDtbsv_v2 ENDP

PUBLIC cublasDtbsv_v2_64
cublasDtbsv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2176]
    jmp rax
cublasDtbsv_v2_64 ENDP

PUBLIC cublasDtpmv
cublasDtpmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2184]
    jmp rax
cublasDtpmv ENDP

PUBLIC cublasDtpmv_v2
cublasDtpmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2192]
    jmp rax
cublasDtpmv_v2 ENDP

PUBLIC cublasDtpmv_v2_64
cublasDtpmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2200]
    jmp rax
cublasDtpmv_v2_64 ENDP

PUBLIC cublasDtpsv
cublasDtpsv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2208]
    jmp rax
cublasDtpsv ENDP

PUBLIC cublasDtpsv_v2
cublasDtpsv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2216]
    jmp rax
cublasDtpsv_v2 ENDP

PUBLIC cublasDtpsv_v2_64
cublasDtpsv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2224]
    jmp rax
cublasDtpsv_v2_64 ENDP

PUBLIC cublasDtpttr
cublasDtpttr PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2232]
    jmp rax
cublasDtpttr ENDP

PUBLIC cublasDtrmm
cublasDtrmm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2240]
    jmp rax
cublasDtrmm ENDP

PUBLIC cublasDtrmm_v2
cublasDtrmm_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2248]
    jmp rax
cublasDtrmm_v2 ENDP

PUBLIC cublasDtrmm_v2_64
cublasDtrmm_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2256]
    jmp rax
cublasDtrmm_v2_64 ENDP

PUBLIC cublasDtrmv
cublasDtrmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2264]
    jmp rax
cublasDtrmv ENDP

PUBLIC cublasDtrmv_v2
cublasDtrmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2272]
    jmp rax
cublasDtrmv_v2 ENDP

PUBLIC cublasDtrmv_v2_64
cublasDtrmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2280]
    jmp rax
cublasDtrmv_v2_64 ENDP

PUBLIC cublasDtrsm
cublasDtrsm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2288]
    jmp rax
cublasDtrsm ENDP

PUBLIC cublasDtrsmBatched
cublasDtrsmBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2296]
    jmp rax
cublasDtrsmBatched ENDP

PUBLIC cublasDtrsmBatched_64
cublasDtrsmBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2304]
    jmp rax
cublasDtrsmBatched_64 ENDP

PUBLIC cublasDtrsm_v2
cublasDtrsm_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2312]
    jmp rax
cublasDtrsm_v2 ENDP

PUBLIC cublasDtrsm_v2_64
cublasDtrsm_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2320]
    jmp rax
cublasDtrsm_v2_64 ENDP

PUBLIC cublasDtrsv
cublasDtrsv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2328]
    jmp rax
cublasDtrsv ENDP

PUBLIC cublasDtrsv_v2
cublasDtrsv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2336]
    jmp rax
cublasDtrsv_v2 ENDP

PUBLIC cublasDtrsv_v2_64
cublasDtrsv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2344]
    jmp rax
cublasDtrsv_v2_64 ENDP

PUBLIC cublasDtrttp
cublasDtrttp PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2352]
    jmp rax
cublasDtrttp ENDP

PUBLIC cublasDzasum
cublasDzasum PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2360]
    jmp rax
cublasDzasum ENDP

PUBLIC cublasDzasum_v2
cublasDzasum_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2368]
    jmp rax
cublasDzasum_v2 ENDP

PUBLIC cublasDzasum_v2_64
cublasDzasum_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2376]
    jmp rax
cublasDzasum_v2_64 ENDP

PUBLIC cublasDznrm2
cublasDznrm2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2384]
    jmp rax
cublasDznrm2 ENDP

PUBLIC cublasDznrm2_v2
cublasDznrm2_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2392]
    jmp rax
cublasDznrm2_v2 ENDP

PUBLIC cublasDznrm2_v2_64
cublasDznrm2_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2400]
    jmp rax
cublasDznrm2_v2_64 ENDP

PUBLIC cublasFree
cublasFree PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2408]
    jmp rax
cublasFree ENDP

PUBLIC cublasGemmBatchedEx
cublasGemmBatchedEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2416]
    jmp rax
cublasGemmBatchedEx ENDP

PUBLIC cublasGemmBatchedEx_64
cublasGemmBatchedEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2424]
    jmp rax
cublasGemmBatchedEx_64 ENDP

PUBLIC cublasGemmEx
cublasGemmEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2432]
    jmp rax
cublasGemmEx ENDP

PUBLIC cublasGemmEx_64
cublasGemmEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2440]
    jmp rax
cublasGemmEx_64 ENDP

PUBLIC cublasGemmGroupedBatchedEx
cublasGemmGroupedBatchedEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2448]
    jmp rax
cublasGemmGroupedBatchedEx ENDP

PUBLIC cublasGemmGroupedBatchedEx_64
cublasGemmGroupedBatchedEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2456]
    jmp rax
cublasGemmGroupedBatchedEx_64 ENDP

PUBLIC cublasGemmStridedBatchedEx
cublasGemmStridedBatchedEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2464]
    jmp rax
cublasGemmStridedBatchedEx ENDP

PUBLIC cublasGemmStridedBatchedEx_64
cublasGemmStridedBatchedEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2472]
    jmp rax
cublasGemmStridedBatchedEx_64 ENDP

PUBLIC cublasGetAtomicsMode
cublasGetAtomicsMode PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2480]
    jmp rax
cublasGetAtomicsMode ENDP

PUBLIC cublasGetBackdoor
cublasGetBackdoor PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2488]
    jmp rax
cublasGetBackdoor ENDP

PUBLIC cublasGetCudartVersion
cublasGetCudartVersion PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2496]
    jmp rax
cublasGetCudartVersion ENDP

PUBLIC cublasGetEmulationStrategy
cublasGetEmulationStrategy PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2504]
    jmp rax
cublasGetEmulationStrategy ENDP

PUBLIC cublasGetEnvironmentMode
cublasGetEnvironmentMode PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2512]
    jmp rax
cublasGetEnvironmentMode ENDP

PUBLIC cublasGetError
cublasGetError PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2520]
    jmp rax
cublasGetError ENDP

PUBLIC cublasGetLoggerCallback
cublasGetLoggerCallback PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2528]
    jmp rax
cublasGetLoggerCallback ENDP

PUBLIC cublasGetMathMode
cublasGetMathMode PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2536]
    jmp rax
cublasGetMathMode ENDP

PUBLIC cublasGetMatrix
cublasGetMatrix PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2544]
    jmp rax
cublasGetMatrix ENDP

PUBLIC cublasGetMatrixAsync
cublasGetMatrixAsync PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2552]
    jmp rax
cublasGetMatrixAsync ENDP

PUBLIC cublasGetMatrixAsync_64
cublasGetMatrixAsync_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2560]
    jmp rax
cublasGetMatrixAsync_64 ENDP

PUBLIC cublasGetMatrix_64
cublasGetMatrix_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2568]
    jmp rax
cublasGetMatrix_64 ENDP

PUBLIC cublasGetPointerMode_v2
cublasGetPointerMode_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2576]
    jmp rax
cublasGetPointerMode_v2 ENDP

PUBLIC cublasGetProperty
cublasGetProperty PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2584]
    jmp rax
cublasGetProperty ENDP

PUBLIC cublasGetSmCountTarget
cublasGetSmCountTarget PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2592]
    jmp rax
cublasGetSmCountTarget ENDP

PUBLIC cublasGetStatusName
cublasGetStatusName PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2600]
    jmp rax
cublasGetStatusName ENDP

PUBLIC cublasGetStatusString
cublasGetStatusString PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2608]
    jmp rax
cublasGetStatusString ENDP

PUBLIC cublasGetStream_v2
cublasGetStream_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2616]
    jmp rax
cublasGetStream_v2 ENDP

PUBLIC cublasGetVector
cublasGetVector PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2624]
    jmp rax
cublasGetVector ENDP

PUBLIC cublasGetVectorAsync
cublasGetVectorAsync PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2632]
    jmp rax
cublasGetVectorAsync ENDP

PUBLIC cublasGetVectorAsync_64
cublasGetVectorAsync_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2640]
    jmp rax
cublasGetVectorAsync_64 ENDP

PUBLIC cublasGetVector_64
cublasGetVector_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2648]
    jmp rax
cublasGetVector_64 ENDP

PUBLIC cublasGetVersion
cublasGetVersion PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2656]
    jmp rax
cublasGetVersion ENDP

PUBLIC cublasGetVersion_v2
cublasGetVersion_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2664]
    jmp rax
cublasGetVersion_v2 ENDP

PUBLIC cublasHSHgemvBatched
cublasHSHgemvBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2672]
    jmp rax
cublasHSHgemvBatched ENDP

PUBLIC cublasHSHgemvBatched_64
cublasHSHgemvBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2680]
    jmp rax
cublasHSHgemvBatched_64 ENDP

PUBLIC cublasHSHgemvStridedBatched
cublasHSHgemvStridedBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2688]
    jmp rax
cublasHSHgemvStridedBatched ENDP

PUBLIC cublasHSHgemvStridedBatched_64
cublasHSHgemvStridedBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2696]
    jmp rax
cublasHSHgemvStridedBatched_64 ENDP

PUBLIC cublasHSSgemvBatched
cublasHSSgemvBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2704]
    jmp rax
cublasHSSgemvBatched ENDP

PUBLIC cublasHSSgemvBatched_64
cublasHSSgemvBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2712]
    jmp rax
cublasHSSgemvBatched_64 ENDP

PUBLIC cublasHSSgemvStridedBatched
cublasHSSgemvStridedBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2720]
    jmp rax
cublasHSSgemvStridedBatched ENDP

PUBLIC cublasHSSgemvStridedBatched_64
cublasHSSgemvStridedBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2728]
    jmp rax
cublasHSSgemvStridedBatched_64 ENDP

PUBLIC cublasHgemm
cublasHgemm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2736]
    jmp rax
cublasHgemm ENDP

PUBLIC cublasHgemmBatched
cublasHgemmBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2744]
    jmp rax
cublasHgemmBatched ENDP

PUBLIC cublasHgemmBatched_64
cublasHgemmBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2752]
    jmp rax
cublasHgemmBatched_64 ENDP

PUBLIC cublasHgemmStridedBatched
cublasHgemmStridedBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2760]
    jmp rax
cublasHgemmStridedBatched ENDP

PUBLIC cublasHgemmStridedBatched_64
cublasHgemmStridedBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2768]
    jmp rax
cublasHgemmStridedBatched_64 ENDP

PUBLIC cublasHgemm_64
cublasHgemm_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2776]
    jmp rax
cublasHgemm_64 ENDP

PUBLIC cublasIamaxEx
cublasIamaxEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2784]
    jmp rax
cublasIamaxEx ENDP

PUBLIC cublasIamaxEx_64
cublasIamaxEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2792]
    jmp rax
cublasIamaxEx_64 ENDP

PUBLIC cublasIaminEx
cublasIaminEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2800]
    jmp rax
cublasIaminEx ENDP

PUBLIC cublasIaminEx_64
cublasIaminEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2808]
    jmp rax
cublasIaminEx_64 ENDP

PUBLIC cublasIcamax
cublasIcamax PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2816]
    jmp rax
cublasIcamax ENDP

PUBLIC cublasIcamax_v2
cublasIcamax_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2824]
    jmp rax
cublasIcamax_v2 ENDP

PUBLIC cublasIcamax_v2_64
cublasIcamax_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2832]
    jmp rax
cublasIcamax_v2_64 ENDP

PUBLIC cublasIcamin
cublasIcamin PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2840]
    jmp rax
cublasIcamin ENDP

PUBLIC cublasIcamin_v2
cublasIcamin_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2848]
    jmp rax
cublasIcamin_v2 ENDP

PUBLIC cublasIcamin_v2_64
cublasIcamin_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2856]
    jmp rax
cublasIcamin_v2_64 ENDP

PUBLIC cublasIdamax
cublasIdamax PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2864]
    jmp rax
cublasIdamax ENDP

PUBLIC cublasIdamax_v2
cublasIdamax_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2872]
    jmp rax
cublasIdamax_v2 ENDP

PUBLIC cublasIdamax_v2_64
cublasIdamax_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2880]
    jmp rax
cublasIdamax_v2_64 ENDP

PUBLIC cublasIdamin
cublasIdamin PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2888]
    jmp rax
cublasIdamin ENDP

PUBLIC cublasIdamin_v2
cublasIdamin_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2896]
    jmp rax
cublasIdamin_v2 ENDP

PUBLIC cublasIdamin_v2_64
cublasIdamin_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2904]
    jmp rax
cublasIdamin_v2_64 ENDP

PUBLIC cublasInit
cublasInit PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2912]
    jmp rax
cublasInit ENDP

PUBLIC cublasIsamax
cublasIsamax PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2920]
    jmp rax
cublasIsamax ENDP

PUBLIC cublasIsamax_v2
cublasIsamax_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2928]
    jmp rax
cublasIsamax_v2 ENDP

PUBLIC cublasIsamax_v2_64
cublasIsamax_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2936]
    jmp rax
cublasIsamax_v2_64 ENDP

PUBLIC cublasIsamin
cublasIsamin PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2944]
    jmp rax
cublasIsamin ENDP

PUBLIC cublasIsamin_v2
cublasIsamin_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2952]
    jmp rax
cublasIsamin_v2 ENDP

PUBLIC cublasIsamin_v2_64
cublasIsamin_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2960]
    jmp rax
cublasIsamin_v2_64 ENDP

PUBLIC cublasIzamax
cublasIzamax PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2968]
    jmp rax
cublasIzamax ENDP

PUBLIC cublasIzamax_v2
cublasIzamax_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2976]
    jmp rax
cublasIzamax_v2 ENDP

PUBLIC cublasIzamax_v2_64
cublasIzamax_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2984]
    jmp rax
cublasIzamax_v2_64 ENDP

PUBLIC cublasIzamin
cublasIzamin PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 2992]
    jmp rax
cublasIzamin ENDP

PUBLIC cublasIzamin_v2
cublasIzamin_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3000]
    jmp rax
cublasIzamin_v2 ENDP

PUBLIC cublasIzamin_v2_64
cublasIzamin_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3008]
    jmp rax
cublasIzamin_v2_64 ENDP

PUBLIC cublasLoggerConfigure
cublasLoggerConfigure PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3016]
    jmp rax
cublasLoggerConfigure ENDP

PUBLIC cublasNrm2Ex
cublasNrm2Ex PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3024]
    jmp rax
cublasNrm2Ex ENDP

PUBLIC cublasNrm2Ex_64
cublasNrm2Ex_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3032]
    jmp rax
cublasNrm2Ex_64 ENDP

PUBLIC cublasRotEx
cublasRotEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3040]
    jmp rax
cublasRotEx ENDP

PUBLIC cublasRotEx_64
cublasRotEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3048]
    jmp rax
cublasRotEx_64 ENDP

PUBLIC cublasRotgEx
cublasRotgEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3056]
    jmp rax
cublasRotgEx ENDP

PUBLIC cublasRotmEx
cublasRotmEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3064]
    jmp rax
cublasRotmEx ENDP

PUBLIC cublasRotmEx_64
cublasRotmEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3072]
    jmp rax
cublasRotmEx_64 ENDP

PUBLIC cublasRotmgEx
cublasRotmgEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3080]
    jmp rax
cublasRotmgEx ENDP

PUBLIC cublasSasum
cublasSasum PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3088]
    jmp rax
cublasSasum ENDP

PUBLIC cublasSasum_v2
cublasSasum_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3096]
    jmp rax
cublasSasum_v2 ENDP

PUBLIC cublasSasum_v2_64
cublasSasum_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3104]
    jmp rax
cublasSasum_v2_64 ENDP

PUBLIC cublasSaxpy
cublasSaxpy PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3112]
    jmp rax
cublasSaxpy ENDP

PUBLIC cublasSaxpy_v2
cublasSaxpy_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3120]
    jmp rax
cublasSaxpy_v2 ENDP

PUBLIC cublasSaxpy_v2_64
cublasSaxpy_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3128]
    jmp rax
cublasSaxpy_v2_64 ENDP

PUBLIC cublasSbdmm
cublasSbdmm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3136]
    jmp rax
cublasSbdmm ENDP

PUBLIC cublasScalEx
cublasScalEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3144]
    jmp rax
cublasScalEx ENDP

PUBLIC cublasScalEx_64
cublasScalEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3152]
    jmp rax
cublasScalEx_64 ENDP

PUBLIC cublasScasum
cublasScasum PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3160]
    jmp rax
cublasScasum ENDP

PUBLIC cublasScasum_v2
cublasScasum_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3168]
    jmp rax
cublasScasum_v2 ENDP

PUBLIC cublasScasum_v2_64
cublasScasum_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3176]
    jmp rax
cublasScasum_v2_64 ENDP

PUBLIC cublasScnrm2
cublasScnrm2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3184]
    jmp rax
cublasScnrm2 ENDP

PUBLIC cublasScnrm2_v2
cublasScnrm2_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3192]
    jmp rax
cublasScnrm2_v2 ENDP

PUBLIC cublasScnrm2_v2_64
cublasScnrm2_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3200]
    jmp rax
cublasScnrm2_v2_64 ENDP

PUBLIC cublasScopy
cublasScopy PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3208]
    jmp rax
cublasScopy ENDP

PUBLIC cublasScopy_v2
cublasScopy_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3216]
    jmp rax
cublasScopy_v2 ENDP

PUBLIC cublasScopy_v2_64
cublasScopy_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3224]
    jmp rax
cublasScopy_v2_64 ENDP

PUBLIC cublasSdgmm
cublasSdgmm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3232]
    jmp rax
cublasSdgmm ENDP

PUBLIC cublasSdgmm_64
cublasSdgmm_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3240]
    jmp rax
cublasSdgmm_64 ENDP

PUBLIC cublasSdot
cublasSdot PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3248]
    jmp rax
cublasSdot ENDP

PUBLIC cublasSdot_v2
cublasSdot_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3256]
    jmp rax
cublasSdot_v2 ENDP

PUBLIC cublasSdot_v2_64
cublasSdot_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3264]
    jmp rax
cublasSdot_v2_64 ENDP

PUBLIC cublasSetAtomicsMode
cublasSetAtomicsMode PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3272]
    jmp rax
cublasSetAtomicsMode ENDP

PUBLIC cublasSetBackdoor
cublasSetBackdoor PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3280]
    jmp rax
cublasSetBackdoor ENDP

PUBLIC cublasSetBackdoorEx
cublasSetBackdoorEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3288]
    jmp rax
cublasSetBackdoorEx ENDP

PUBLIC cublasSetEmulationStrategy
cublasSetEmulationStrategy PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3296]
    jmp rax
cublasSetEmulationStrategy ENDP

PUBLIC cublasSetEnvironmentMode
cublasSetEnvironmentMode PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3304]
    jmp rax
cublasSetEnvironmentMode ENDP

PUBLIC cublasSetKernelStream
cublasSetKernelStream PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3312]
    jmp rax
cublasSetKernelStream ENDP

PUBLIC cublasSetLoggerCallback
cublasSetLoggerCallback PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3320]
    jmp rax
cublasSetLoggerCallback ENDP

PUBLIC cublasSetMathMode
cublasSetMathMode PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3328]
    jmp rax
cublasSetMathMode ENDP

PUBLIC cublasSetMatrix
cublasSetMatrix PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3336]
    jmp rax
cublasSetMatrix ENDP

PUBLIC cublasSetMatrixAsync
cublasSetMatrixAsync PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3344]
    jmp rax
cublasSetMatrixAsync ENDP

PUBLIC cublasSetMatrixAsync_64
cublasSetMatrixAsync_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3352]
    jmp rax
cublasSetMatrixAsync_64 ENDP

PUBLIC cublasSetMatrix_64
cublasSetMatrix_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3360]
    jmp rax
cublasSetMatrix_64 ENDP

PUBLIC cublasSetPointerMode_v2
cublasSetPointerMode_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3368]
    jmp rax
cublasSetPointerMode_v2 ENDP

PUBLIC cublasSetSmCountTarget
cublasSetSmCountTarget PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3376]
    jmp rax
cublasSetSmCountTarget ENDP

PUBLIC cublasSetStream_v2
cublasSetStream_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3384]
    jmp rax
cublasSetStream_v2 ENDP

PUBLIC cublasSetVector
cublasSetVector PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3392]
    jmp rax
cublasSetVector ENDP

PUBLIC cublasSetVectorAsync
cublasSetVectorAsync PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3400]
    jmp rax
cublasSetVectorAsync ENDP

PUBLIC cublasSetVectorAsync_64
cublasSetVectorAsync_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3408]
    jmp rax
cublasSetVectorAsync_64 ENDP

PUBLIC cublasSetVector_64
cublasSetVector_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3416]
    jmp rax
cublasSetVector_64 ENDP

PUBLIC cublasSetWorkspace_v2
cublasSetWorkspace_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3424]
    jmp rax
cublasSetWorkspace_v2 ENDP

PUBLIC cublasSgbmv
cublasSgbmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3432]
    jmp rax
cublasSgbmv ENDP

PUBLIC cublasSgbmv_v2
cublasSgbmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3440]
    jmp rax
cublasSgbmv_v2 ENDP

PUBLIC cublasSgbmv_v2_64
cublasSgbmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3448]
    jmp rax
cublasSgbmv_v2_64 ENDP

PUBLIC cublasSgeam
cublasSgeam PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3456]
    jmp rax
cublasSgeam ENDP

PUBLIC cublasSgeam_64
cublasSgeam_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3464]
    jmp rax
cublasSgeam_64 ENDP

PUBLIC cublasSgelsBatched
cublasSgelsBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3472]
    jmp rax
cublasSgelsBatched ENDP

PUBLIC cublasSgemm
cublasSgemm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3480]
    jmp rax
cublasSgemm ENDP

PUBLIC cublasSgemmBatched
cublasSgemmBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3488]
    jmp rax
cublasSgemmBatched ENDP

PUBLIC cublasSgemmBatched_64
cublasSgemmBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3496]
    jmp rax
cublasSgemmBatched_64 ENDP

PUBLIC cublasSgemmEx
cublasSgemmEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3504]
    jmp rax
cublasSgemmEx ENDP

PUBLIC cublasSgemmEx_64
cublasSgemmEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3512]
    jmp rax
cublasSgemmEx_64 ENDP

PUBLIC cublasSgemmGroupedBatched
cublasSgemmGroupedBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3520]
    jmp rax
cublasSgemmGroupedBatched ENDP

PUBLIC cublasSgemmGroupedBatched_64
cublasSgemmGroupedBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3528]
    jmp rax
cublasSgemmGroupedBatched_64 ENDP

PUBLIC cublasSgemmStridedBatched
cublasSgemmStridedBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3536]
    jmp rax
cublasSgemmStridedBatched ENDP

PUBLIC cublasSgemmStridedBatched_64
cublasSgemmStridedBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3544]
    jmp rax
cublasSgemmStridedBatched_64 ENDP

PUBLIC cublasSgemm_v2
cublasSgemm_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3552]
    jmp rax
cublasSgemm_v2 ENDP

PUBLIC cublasSgemm_v2_64
cublasSgemm_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3560]
    jmp rax
cublasSgemm_v2_64 ENDP

PUBLIC cublasSgemv
cublasSgemv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3568]
    jmp rax
cublasSgemv ENDP

PUBLIC cublasSgemvBatched
cublasSgemvBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3576]
    jmp rax
cublasSgemvBatched ENDP

PUBLIC cublasSgemvBatched_64
cublasSgemvBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3584]
    jmp rax
cublasSgemvBatched_64 ENDP

PUBLIC cublasSgemvStridedBatched
cublasSgemvStridedBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3592]
    jmp rax
cublasSgemvStridedBatched ENDP

PUBLIC cublasSgemvStridedBatched_64
cublasSgemvStridedBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3600]
    jmp rax
cublasSgemvStridedBatched_64 ENDP

PUBLIC cublasSgemv_v2
cublasSgemv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3608]
    jmp rax
cublasSgemv_v2 ENDP

PUBLIC cublasSgemv_v2_64
cublasSgemv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3616]
    jmp rax
cublasSgemv_v2_64 ENDP

PUBLIC cublasSgeqrfBatched
cublasSgeqrfBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3624]
    jmp rax
cublasSgeqrfBatched ENDP

PUBLIC cublasSger
cublasSger PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3632]
    jmp rax
cublasSger ENDP

PUBLIC cublasSger_v2
cublasSger_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3640]
    jmp rax
cublasSger_v2 ENDP

PUBLIC cublasSger_v2_64
cublasSger_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3648]
    jmp rax
cublasSger_v2_64 ENDP

PUBLIC cublasSgetrfBatched
cublasSgetrfBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3656]
    jmp rax
cublasSgetrfBatched ENDP

PUBLIC cublasSgetriBatched
cublasSgetriBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3664]
    jmp rax
cublasSgetriBatched ENDP

PUBLIC cublasSgetrsBatched
cublasSgetrsBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3672]
    jmp rax
cublasSgetrsBatched ENDP

PUBLIC cublasShutdown
cublasShutdown PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3680]
    jmp rax
cublasShutdown ENDP

PUBLIC cublasSmatinvBatched
cublasSmatinvBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3688]
    jmp rax
cublasSmatinvBatched ENDP

PUBLIC cublasSnrm2
cublasSnrm2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3696]
    jmp rax
cublasSnrm2 ENDP

PUBLIC cublasSnrm2_v2
cublasSnrm2_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3704]
    jmp rax
cublasSnrm2_v2 ENDP

PUBLIC cublasSnrm2_v2_64
cublasSnrm2_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3712]
    jmp rax
cublasSnrm2_v2_64 ENDP

PUBLIC cublasSrot
cublasSrot PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3720]
    jmp rax
cublasSrot ENDP

PUBLIC cublasSrot_v2
cublasSrot_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3728]
    jmp rax
cublasSrot_v2 ENDP

PUBLIC cublasSrot_v2_64
cublasSrot_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3736]
    jmp rax
cublasSrot_v2_64 ENDP

PUBLIC cublasSrotg
cublasSrotg PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3744]
    jmp rax
cublasSrotg ENDP

PUBLIC cublasSrotg_v2
cublasSrotg_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3752]
    jmp rax
cublasSrotg_v2 ENDP

PUBLIC cublasSrotm
cublasSrotm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3760]
    jmp rax
cublasSrotm ENDP

PUBLIC cublasSrotm_v2
cublasSrotm_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3768]
    jmp rax
cublasSrotm_v2 ENDP

PUBLIC cublasSrotm_v2_64
cublasSrotm_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3776]
    jmp rax
cublasSrotm_v2_64 ENDP

PUBLIC cublasSrotmg
cublasSrotmg PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3784]
    jmp rax
cublasSrotmg ENDP

PUBLIC cublasSrotmg_v2
cublasSrotmg_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3792]
    jmp rax
cublasSrotmg_v2 ENDP

PUBLIC cublasSsbmv
cublasSsbmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3800]
    jmp rax
cublasSsbmv ENDP

PUBLIC cublasSsbmv_v2
cublasSsbmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3808]
    jmp rax
cublasSsbmv_v2 ENDP

PUBLIC cublasSsbmv_v2_64
cublasSsbmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3816]
    jmp rax
cublasSsbmv_v2_64 ENDP

PUBLIC cublasSscal
cublasSscal PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3824]
    jmp rax
cublasSscal ENDP

PUBLIC cublasSscal_v2
cublasSscal_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3832]
    jmp rax
cublasSscal_v2 ENDP

PUBLIC cublasSscal_v2_64
cublasSscal_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3840]
    jmp rax
cublasSscal_v2_64 ENDP

PUBLIC cublasSspmv
cublasSspmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3848]
    jmp rax
cublasSspmv ENDP

PUBLIC cublasSspmv_v2
cublasSspmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3856]
    jmp rax
cublasSspmv_v2 ENDP

PUBLIC cublasSspmv_v2_64
cublasSspmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3864]
    jmp rax
cublasSspmv_v2_64 ENDP

PUBLIC cublasSspr
cublasSspr PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3872]
    jmp rax
cublasSspr ENDP

PUBLIC cublasSspr2
cublasSspr2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3880]
    jmp rax
cublasSspr2 ENDP

PUBLIC cublasSspr2_v2
cublasSspr2_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3888]
    jmp rax
cublasSspr2_v2 ENDP

PUBLIC cublasSspr2_v2_64
cublasSspr2_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3896]
    jmp rax
cublasSspr2_v2_64 ENDP

PUBLIC cublasSspr_v2
cublasSspr_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3904]
    jmp rax
cublasSspr_v2 ENDP

PUBLIC cublasSspr_v2_64
cublasSspr_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3912]
    jmp rax
cublasSspr_v2_64 ENDP

PUBLIC cublasSswap
cublasSswap PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3920]
    jmp rax
cublasSswap ENDP

PUBLIC cublasSswap_v2
cublasSswap_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3928]
    jmp rax
cublasSswap_v2 ENDP

PUBLIC cublasSswap_v2_64
cublasSswap_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3936]
    jmp rax
cublasSswap_v2_64 ENDP

PUBLIC cublasSsymm
cublasSsymm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3944]
    jmp rax
cublasSsymm ENDP

PUBLIC cublasSsymm_v2
cublasSsymm_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3952]
    jmp rax
cublasSsymm_v2 ENDP

PUBLIC cublasSsymm_v2_64
cublasSsymm_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3960]
    jmp rax
cublasSsymm_v2_64 ENDP

PUBLIC cublasSsymv
cublasSsymv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3968]
    jmp rax
cublasSsymv ENDP

PUBLIC cublasSsymv_v2
cublasSsymv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3976]
    jmp rax
cublasSsymv_v2 ENDP

PUBLIC cublasSsymv_v2_64
cublasSsymv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3984]
    jmp rax
cublasSsymv_v2_64 ENDP

PUBLIC cublasSsyr
cublasSsyr PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 3992]
    jmp rax
cublasSsyr ENDP

PUBLIC cublasSsyr2
cublasSsyr2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4000]
    jmp rax
cublasSsyr2 ENDP

PUBLIC cublasSsyr2_v2
cublasSsyr2_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4008]
    jmp rax
cublasSsyr2_v2 ENDP

PUBLIC cublasSsyr2_v2_64
cublasSsyr2_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4016]
    jmp rax
cublasSsyr2_v2_64 ENDP

PUBLIC cublasSsyr2k
cublasSsyr2k PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4024]
    jmp rax
cublasSsyr2k ENDP

PUBLIC cublasSsyr2k_v2
cublasSsyr2k_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4032]
    jmp rax
cublasSsyr2k_v2 ENDP

PUBLIC cublasSsyr2k_v2_64
cublasSsyr2k_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4040]
    jmp rax
cublasSsyr2k_v2_64 ENDP

PUBLIC cublasSsyr_v2
cublasSsyr_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4048]
    jmp rax
cublasSsyr_v2 ENDP

PUBLIC cublasSsyr_v2_64
cublasSsyr_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4056]
    jmp rax
cublasSsyr_v2_64 ENDP

PUBLIC cublasSsyrk
cublasSsyrk PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4064]
    jmp rax
cublasSsyrk ENDP

PUBLIC cublasSsyrk_v2
cublasSsyrk_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4072]
    jmp rax
cublasSsyrk_v2 ENDP

PUBLIC cublasSsyrk_v2_64
cublasSsyrk_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4080]
    jmp rax
cublasSsyrk_v2_64 ENDP

PUBLIC cublasSsyrkx
cublasSsyrkx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4088]
    jmp rax
cublasSsyrkx ENDP

PUBLIC cublasSsyrkx_64
cublasSsyrkx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4096]
    jmp rax
cublasSsyrkx_64 ENDP

PUBLIC cublasStbmv
cublasStbmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4104]
    jmp rax
cublasStbmv ENDP

PUBLIC cublasStbmv_v2
cublasStbmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4112]
    jmp rax
cublasStbmv_v2 ENDP

PUBLIC cublasStbmv_v2_64
cublasStbmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4120]
    jmp rax
cublasStbmv_v2_64 ENDP

PUBLIC cublasStbsv
cublasStbsv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4128]
    jmp rax
cublasStbsv ENDP

PUBLIC cublasStbsv_v2
cublasStbsv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4136]
    jmp rax
cublasStbsv_v2 ENDP

PUBLIC cublasStbsv_v2_64
cublasStbsv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4144]
    jmp rax
cublasStbsv_v2_64 ENDP

PUBLIC cublasStpmv
cublasStpmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4152]
    jmp rax
cublasStpmv ENDP

PUBLIC cublasStpmv_v2
cublasStpmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4160]
    jmp rax
cublasStpmv_v2 ENDP

PUBLIC cublasStpmv_v2_64
cublasStpmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4168]
    jmp rax
cublasStpmv_v2_64 ENDP

PUBLIC cublasStpsv
cublasStpsv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4176]
    jmp rax
cublasStpsv ENDP

PUBLIC cublasStpsv_v2
cublasStpsv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4184]
    jmp rax
cublasStpsv_v2 ENDP

PUBLIC cublasStpsv_v2_64
cublasStpsv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4192]
    jmp rax
cublasStpsv_v2_64 ENDP

PUBLIC cublasStpttr
cublasStpttr PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4200]
    jmp rax
cublasStpttr ENDP

PUBLIC cublasStrmm
cublasStrmm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4208]
    jmp rax
cublasStrmm ENDP

PUBLIC cublasStrmm_v2
cublasStrmm_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4216]
    jmp rax
cublasStrmm_v2 ENDP

PUBLIC cublasStrmm_v2_64
cublasStrmm_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4224]
    jmp rax
cublasStrmm_v2_64 ENDP

PUBLIC cublasStrmv
cublasStrmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4232]
    jmp rax
cublasStrmv ENDP

PUBLIC cublasStrmv_v2
cublasStrmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4240]
    jmp rax
cublasStrmv_v2 ENDP

PUBLIC cublasStrmv_v2_64
cublasStrmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4248]
    jmp rax
cublasStrmv_v2_64 ENDP

PUBLIC cublasStrsm
cublasStrsm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4256]
    jmp rax
cublasStrsm ENDP

PUBLIC cublasStrsmBatched
cublasStrsmBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4264]
    jmp rax
cublasStrsmBatched ENDP

PUBLIC cublasStrsmBatched_64
cublasStrsmBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4272]
    jmp rax
cublasStrsmBatched_64 ENDP

PUBLIC cublasStrsm_v2
cublasStrsm_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4280]
    jmp rax
cublasStrsm_v2 ENDP

PUBLIC cublasStrsm_v2_64
cublasStrsm_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4288]
    jmp rax
cublasStrsm_v2_64 ENDP

PUBLIC cublasStrsv
cublasStrsv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4296]
    jmp rax
cublasStrsv ENDP

PUBLIC cublasStrsv_v2
cublasStrsv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4304]
    jmp rax
cublasStrsv_v2 ENDP

PUBLIC cublasStrsv_v2_64
cublasStrsv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4312]
    jmp rax
cublasStrsv_v2_64 ENDP

PUBLIC cublasStrttp
cublasStrttp PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4320]
    jmp rax
cublasStrttp ENDP

PUBLIC cublasSwapEx
cublasSwapEx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4328]
    jmp rax
cublasSwapEx ENDP

PUBLIC cublasSwapEx_64
cublasSwapEx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4336]
    jmp rax
cublasSwapEx_64 ENDP

PUBLIC cublasTSSgemvBatched
cublasTSSgemvBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4344]
    jmp rax
cublasTSSgemvBatched ENDP

PUBLIC cublasTSSgemvBatched_64
cublasTSSgemvBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4352]
    jmp rax
cublasTSSgemvBatched_64 ENDP

PUBLIC cublasTSSgemvStridedBatched
cublasTSSgemvStridedBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4360]
    jmp rax
cublasTSSgemvStridedBatched ENDP

PUBLIC cublasTSSgemvStridedBatched_64
cublasTSSgemvStridedBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4368]
    jmp rax
cublasTSSgemvStridedBatched_64 ENDP

PUBLIC cublasTSTgemvBatched
cublasTSTgemvBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4376]
    jmp rax
cublasTSTgemvBatched ENDP

PUBLIC cublasTSTgemvBatched_64
cublasTSTgemvBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4384]
    jmp rax
cublasTSTgemvBatched_64 ENDP

PUBLIC cublasTSTgemvStridedBatched
cublasTSTgemvStridedBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4392]
    jmp rax
cublasTSTgemvStridedBatched ENDP

PUBLIC cublasTSTgemvStridedBatched_64
cublasTSTgemvStridedBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4400]
    jmp rax
cublasTSTgemvStridedBatched_64 ENDP

PUBLIC cublasUint8gemmBias
cublasUint8gemmBias PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4408]
    jmp rax
cublasUint8gemmBias ENDP

PUBLIC cublasXerbla
cublasXerbla PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4416]
    jmp rax
cublasXerbla ENDP

PUBLIC cublasXtCgemm
cublasXtCgemm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4424]
    jmp rax
cublasXtCgemm ENDP

PUBLIC cublasXtChemm
cublasXtChemm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4432]
    jmp rax
cublasXtChemm ENDP

PUBLIC cublasXtCher2k
cublasXtCher2k PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4440]
    jmp rax
cublasXtCher2k ENDP

PUBLIC cublasXtCherk
cublasXtCherk PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4448]
    jmp rax
cublasXtCherk ENDP

PUBLIC cublasXtCherkx
cublasXtCherkx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4456]
    jmp rax
cublasXtCherkx ENDP

PUBLIC cublasXtCreate
cublasXtCreate PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4464]
    jmp rax
cublasXtCreate ENDP

PUBLIC cublasXtCspmm
cublasXtCspmm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4472]
    jmp rax
cublasXtCspmm ENDP

PUBLIC cublasXtCsymm
cublasXtCsymm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4480]
    jmp rax
cublasXtCsymm ENDP

PUBLIC cublasXtCsyr2k
cublasXtCsyr2k PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4488]
    jmp rax
cublasXtCsyr2k ENDP

PUBLIC cublasXtCsyrk
cublasXtCsyrk PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4496]
    jmp rax
cublasXtCsyrk ENDP

PUBLIC cublasXtCsyrkx
cublasXtCsyrkx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4504]
    jmp rax
cublasXtCsyrkx ENDP

PUBLIC cublasXtCtrmm
cublasXtCtrmm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4512]
    jmp rax
cublasXtCtrmm ENDP

PUBLIC cublasXtCtrsm
cublasXtCtrsm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4520]
    jmp rax
cublasXtCtrsm ENDP

PUBLIC cublasXtDestroy
cublasXtDestroy PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4528]
    jmp rax
cublasXtDestroy ENDP

PUBLIC cublasXtDeviceSelect
cublasXtDeviceSelect PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4536]
    jmp rax
cublasXtDeviceSelect ENDP

PUBLIC cublasXtDgemm
cublasXtDgemm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4544]
    jmp rax
cublasXtDgemm ENDP

PUBLIC cublasXtDspmm
cublasXtDspmm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4552]
    jmp rax
cublasXtDspmm ENDP

PUBLIC cublasXtDsymm
cublasXtDsymm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4560]
    jmp rax
cublasXtDsymm ENDP

PUBLIC cublasXtDsyr2k
cublasXtDsyr2k PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4568]
    jmp rax
cublasXtDsyr2k ENDP

PUBLIC cublasXtDsyrk
cublasXtDsyrk PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4576]
    jmp rax
cublasXtDsyrk ENDP

PUBLIC cublasXtDsyrkx
cublasXtDsyrkx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4584]
    jmp rax
cublasXtDsyrkx ENDP

PUBLIC cublasXtDtrmm
cublasXtDtrmm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4592]
    jmp rax
cublasXtDtrmm ENDP

PUBLIC cublasXtDtrsm
cublasXtDtrsm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4600]
    jmp rax
cublasXtDtrsm ENDP

PUBLIC cublasXtGetBlockDim
cublasXtGetBlockDim PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4608]
    jmp rax
cublasXtGetBlockDim ENDP

PUBLIC cublasXtGetNumBoards
cublasXtGetNumBoards PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4616]
    jmp rax
cublasXtGetNumBoards ENDP

PUBLIC cublasXtGetPinningMemMode
cublasXtGetPinningMemMode PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4624]
    jmp rax
cublasXtGetPinningMemMode ENDP

PUBLIC cublasXtMaxBoards
cublasXtMaxBoards PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4632]
    jmp rax
cublasXtMaxBoards ENDP

PUBLIC cublasXtSetBlockDim
cublasXtSetBlockDim PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4640]
    jmp rax
cublasXtSetBlockDim ENDP

PUBLIC cublasXtSetCpuRatio
cublasXtSetCpuRatio PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4648]
    jmp rax
cublasXtSetCpuRatio ENDP

PUBLIC cublasXtSetCpuRoutine
cublasXtSetCpuRoutine PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4656]
    jmp rax
cublasXtSetCpuRoutine ENDP

PUBLIC cublasXtSetPinningMemMode
cublasXtSetPinningMemMode PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4664]
    jmp rax
cublasXtSetPinningMemMode ENDP

PUBLIC cublasXtSgemm
cublasXtSgemm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4672]
    jmp rax
cublasXtSgemm ENDP

PUBLIC cublasXtSspmm
cublasXtSspmm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4680]
    jmp rax
cublasXtSspmm ENDP

PUBLIC cublasXtSsymm
cublasXtSsymm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4688]
    jmp rax
cublasXtSsymm ENDP

PUBLIC cublasXtSsyr2k
cublasXtSsyr2k PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4696]
    jmp rax
cublasXtSsyr2k ENDP

PUBLIC cublasXtSsyrk
cublasXtSsyrk PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4704]
    jmp rax
cublasXtSsyrk ENDP

PUBLIC cublasXtSsyrkx
cublasXtSsyrkx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4712]
    jmp rax
cublasXtSsyrkx ENDP

PUBLIC cublasXtStrmm
cublasXtStrmm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4720]
    jmp rax
cublasXtStrmm ENDP

PUBLIC cublasXtStrsm
cublasXtStrsm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4728]
    jmp rax
cublasXtStrsm ENDP

PUBLIC cublasXtZgemm
cublasXtZgemm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4736]
    jmp rax
cublasXtZgemm ENDP

PUBLIC cublasXtZhemm
cublasXtZhemm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4744]
    jmp rax
cublasXtZhemm ENDP

PUBLIC cublasXtZher2k
cublasXtZher2k PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4752]
    jmp rax
cublasXtZher2k ENDP

PUBLIC cublasXtZherk
cublasXtZherk PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4760]
    jmp rax
cublasXtZherk ENDP

PUBLIC cublasXtZherkx
cublasXtZherkx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4768]
    jmp rax
cublasXtZherkx ENDP

PUBLIC cublasXtZspmm
cublasXtZspmm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4776]
    jmp rax
cublasXtZspmm ENDP

PUBLIC cublasXtZsymm
cublasXtZsymm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4784]
    jmp rax
cublasXtZsymm ENDP

PUBLIC cublasXtZsyr2k
cublasXtZsyr2k PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4792]
    jmp rax
cublasXtZsyr2k ENDP

PUBLIC cublasXtZsyrk
cublasXtZsyrk PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4800]
    jmp rax
cublasXtZsyrk ENDP

PUBLIC cublasXtZsyrkx
cublasXtZsyrkx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4808]
    jmp rax
cublasXtZsyrkx ENDP

PUBLIC cublasXtZtrmm
cublasXtZtrmm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4816]
    jmp rax
cublasXtZtrmm ENDP

PUBLIC cublasXtZtrsm
cublasXtZtrsm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4824]
    jmp rax
cublasXtZtrsm ENDP

PUBLIC cublasZaxpy
cublasZaxpy PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4832]
    jmp rax
cublasZaxpy ENDP

PUBLIC cublasZaxpy_v2
cublasZaxpy_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4840]
    jmp rax
cublasZaxpy_v2 ENDP

PUBLIC cublasZaxpy_v2_64
cublasZaxpy_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4848]
    jmp rax
cublasZaxpy_v2_64 ENDP

PUBLIC cublasZbdmm
cublasZbdmm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4856]
    jmp rax
cublasZbdmm ENDP

PUBLIC cublasZcopy
cublasZcopy PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4864]
    jmp rax
cublasZcopy ENDP

PUBLIC cublasZcopy_v2
cublasZcopy_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4872]
    jmp rax
cublasZcopy_v2 ENDP

PUBLIC cublasZcopy_v2_64
cublasZcopy_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4880]
    jmp rax
cublasZcopy_v2_64 ENDP

PUBLIC cublasZdgmm
cublasZdgmm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4888]
    jmp rax
cublasZdgmm ENDP

PUBLIC cublasZdgmm_64
cublasZdgmm_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4896]
    jmp rax
cublasZdgmm_64 ENDP

PUBLIC cublasZdotc
cublasZdotc PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4904]
    jmp rax
cublasZdotc ENDP

PUBLIC cublasZdotc_v2
cublasZdotc_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4912]
    jmp rax
cublasZdotc_v2 ENDP

PUBLIC cublasZdotc_v2_64
cublasZdotc_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4920]
    jmp rax
cublasZdotc_v2_64 ENDP

PUBLIC cublasZdotu
cublasZdotu PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4928]
    jmp rax
cublasZdotu ENDP

PUBLIC cublasZdotu_v2
cublasZdotu_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4936]
    jmp rax
cublasZdotu_v2 ENDP

PUBLIC cublasZdotu_v2_64
cublasZdotu_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4944]
    jmp rax
cublasZdotu_v2_64 ENDP

PUBLIC cublasZdrot
cublasZdrot PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4952]
    jmp rax
cublasZdrot ENDP

PUBLIC cublasZdrot_v2
cublasZdrot_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4960]
    jmp rax
cublasZdrot_v2 ENDP

PUBLIC cublasZdrot_v2_64
cublasZdrot_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4968]
    jmp rax
cublasZdrot_v2_64 ENDP

PUBLIC cublasZdscal
cublasZdscal PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4976]
    jmp rax
cublasZdscal ENDP

PUBLIC cublasZdscal_v2
cublasZdscal_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4984]
    jmp rax
cublasZdscal_v2 ENDP

PUBLIC cublasZdscal_v2_64
cublasZdscal_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 4992]
    jmp rax
cublasZdscal_v2_64 ENDP

PUBLIC cublasZgbmv
cublasZgbmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5000]
    jmp rax
cublasZgbmv ENDP

PUBLIC cublasZgbmv_v2
cublasZgbmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5008]
    jmp rax
cublasZgbmv_v2 ENDP

PUBLIC cublasZgbmv_v2_64
cublasZgbmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5016]
    jmp rax
cublasZgbmv_v2_64 ENDP

PUBLIC cublasZgeam
cublasZgeam PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5024]
    jmp rax
cublasZgeam ENDP

PUBLIC cublasZgeam_64
cublasZgeam_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5032]
    jmp rax
cublasZgeam_64 ENDP

PUBLIC cublasZgelsBatched
cublasZgelsBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5040]
    jmp rax
cublasZgelsBatched ENDP

PUBLIC cublasZgemm
cublasZgemm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5048]
    jmp rax
cublasZgemm ENDP

PUBLIC cublasZgemm3m
cublasZgemm3m PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5056]
    jmp rax
cublasZgemm3m ENDP

PUBLIC cublasZgemm3m_64
cublasZgemm3m_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5064]
    jmp rax
cublasZgemm3m_64 ENDP

PUBLIC cublasZgemmBatched
cublasZgemmBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5072]
    jmp rax
cublasZgemmBatched ENDP

PUBLIC cublasZgemmBatched_64
cublasZgemmBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5080]
    jmp rax
cublasZgemmBatched_64 ENDP

PUBLIC cublasZgemmStridedBatched
cublasZgemmStridedBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5088]
    jmp rax
cublasZgemmStridedBatched ENDP

PUBLIC cublasZgemmStridedBatched_64
cublasZgemmStridedBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5096]
    jmp rax
cublasZgemmStridedBatched_64 ENDP

PUBLIC cublasZgemm_v2
cublasZgemm_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5104]
    jmp rax
cublasZgemm_v2 ENDP

PUBLIC cublasZgemm_v2_64
cublasZgemm_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5112]
    jmp rax
cublasZgemm_v2_64 ENDP

PUBLIC cublasZgemv
cublasZgemv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5120]
    jmp rax
cublasZgemv ENDP

PUBLIC cublasZgemvBatched
cublasZgemvBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5128]
    jmp rax
cublasZgemvBatched ENDP

PUBLIC cublasZgemvBatched_64
cublasZgemvBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5136]
    jmp rax
cublasZgemvBatched_64 ENDP

PUBLIC cublasZgemvStridedBatched
cublasZgemvStridedBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5144]
    jmp rax
cublasZgemvStridedBatched ENDP

PUBLIC cublasZgemvStridedBatched_64
cublasZgemvStridedBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5152]
    jmp rax
cublasZgemvStridedBatched_64 ENDP

PUBLIC cublasZgemv_v2
cublasZgemv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5160]
    jmp rax
cublasZgemv_v2 ENDP

PUBLIC cublasZgemv_v2_64
cublasZgemv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5168]
    jmp rax
cublasZgemv_v2_64 ENDP

PUBLIC cublasZgeqrfBatched
cublasZgeqrfBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5176]
    jmp rax
cublasZgeqrfBatched ENDP

PUBLIC cublasZgerc
cublasZgerc PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5184]
    jmp rax
cublasZgerc ENDP

PUBLIC cublasZgerc_v2
cublasZgerc_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5192]
    jmp rax
cublasZgerc_v2 ENDP

PUBLIC cublasZgerc_v2_64
cublasZgerc_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5200]
    jmp rax
cublasZgerc_v2_64 ENDP

PUBLIC cublasZgeru
cublasZgeru PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5208]
    jmp rax
cublasZgeru ENDP

PUBLIC cublasZgeru_v2
cublasZgeru_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5216]
    jmp rax
cublasZgeru_v2 ENDP

PUBLIC cublasZgeru_v2_64
cublasZgeru_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5224]
    jmp rax
cublasZgeru_v2_64 ENDP

PUBLIC cublasZgetrfBatched
cublasZgetrfBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5232]
    jmp rax
cublasZgetrfBatched ENDP

PUBLIC cublasZgetriBatched
cublasZgetriBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5240]
    jmp rax
cublasZgetriBatched ENDP

PUBLIC cublasZgetrsBatched
cublasZgetrsBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5248]
    jmp rax
cublasZgetrsBatched ENDP

PUBLIC cublasZhbmv
cublasZhbmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5256]
    jmp rax
cublasZhbmv ENDP

PUBLIC cublasZhbmv_v2
cublasZhbmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5264]
    jmp rax
cublasZhbmv_v2 ENDP

PUBLIC cublasZhbmv_v2_64
cublasZhbmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5272]
    jmp rax
cublasZhbmv_v2_64 ENDP

PUBLIC cublasZhemm
cublasZhemm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5280]
    jmp rax
cublasZhemm ENDP

PUBLIC cublasZhemm_v2
cublasZhemm_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5288]
    jmp rax
cublasZhemm_v2 ENDP

PUBLIC cublasZhemm_v2_64
cublasZhemm_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5296]
    jmp rax
cublasZhemm_v2_64 ENDP

PUBLIC cublasZhemv
cublasZhemv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5304]
    jmp rax
cublasZhemv ENDP

PUBLIC cublasZhemv_v2
cublasZhemv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5312]
    jmp rax
cublasZhemv_v2 ENDP

PUBLIC cublasZhemv_v2_64
cublasZhemv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5320]
    jmp rax
cublasZhemv_v2_64 ENDP

PUBLIC cublasZher
cublasZher PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5328]
    jmp rax
cublasZher ENDP

PUBLIC cublasZher2
cublasZher2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5336]
    jmp rax
cublasZher2 ENDP

PUBLIC cublasZher2_v2
cublasZher2_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5344]
    jmp rax
cublasZher2_v2 ENDP

PUBLIC cublasZher2_v2_64
cublasZher2_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5352]
    jmp rax
cublasZher2_v2_64 ENDP

PUBLIC cublasZher2k
cublasZher2k PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5360]
    jmp rax
cublasZher2k ENDP

PUBLIC cublasZher2k_v2
cublasZher2k_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5368]
    jmp rax
cublasZher2k_v2 ENDP

PUBLIC cublasZher2k_v2_64
cublasZher2k_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5376]
    jmp rax
cublasZher2k_v2_64 ENDP

PUBLIC cublasZher_v2
cublasZher_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5384]
    jmp rax
cublasZher_v2 ENDP

PUBLIC cublasZher_v2_64
cublasZher_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5392]
    jmp rax
cublasZher_v2_64 ENDP

PUBLIC cublasZherk
cublasZherk PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5400]
    jmp rax
cublasZherk ENDP

PUBLIC cublasZherk_v2
cublasZherk_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5408]
    jmp rax
cublasZherk_v2 ENDP

PUBLIC cublasZherk_v2_64
cublasZherk_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5416]
    jmp rax
cublasZherk_v2_64 ENDP

PUBLIC cublasZherkx
cublasZherkx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5424]
    jmp rax
cublasZherkx ENDP

PUBLIC cublasZherkx_64
cublasZherkx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5432]
    jmp rax
cublasZherkx_64 ENDP

PUBLIC cublasZhpmv
cublasZhpmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5440]
    jmp rax
cublasZhpmv ENDP

PUBLIC cublasZhpmv_v2
cublasZhpmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5448]
    jmp rax
cublasZhpmv_v2 ENDP

PUBLIC cublasZhpmv_v2_64
cublasZhpmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5456]
    jmp rax
cublasZhpmv_v2_64 ENDP

PUBLIC cublasZhpr
cublasZhpr PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5464]
    jmp rax
cublasZhpr ENDP

PUBLIC cublasZhpr2
cublasZhpr2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5472]
    jmp rax
cublasZhpr2 ENDP

PUBLIC cublasZhpr2_v2
cublasZhpr2_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5480]
    jmp rax
cublasZhpr2_v2 ENDP

PUBLIC cublasZhpr2_v2_64
cublasZhpr2_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5488]
    jmp rax
cublasZhpr2_v2_64 ENDP

PUBLIC cublasZhpr_v2
cublasZhpr_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5496]
    jmp rax
cublasZhpr_v2 ENDP

PUBLIC cublasZhpr_v2_64
cublasZhpr_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5504]
    jmp rax
cublasZhpr_v2_64 ENDP

PUBLIC cublasZmatinvBatched
cublasZmatinvBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5512]
    jmp rax
cublasZmatinvBatched ENDP

PUBLIC cublasZrot
cublasZrot PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5520]
    jmp rax
cublasZrot ENDP

PUBLIC cublasZrot_v2
cublasZrot_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5528]
    jmp rax
cublasZrot_v2 ENDP

PUBLIC cublasZrot_v2_64
cublasZrot_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5536]
    jmp rax
cublasZrot_v2_64 ENDP

PUBLIC cublasZrotg
cublasZrotg PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5544]
    jmp rax
cublasZrotg ENDP

PUBLIC cublasZrotg_v2
cublasZrotg_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5552]
    jmp rax
cublasZrotg_v2 ENDP

PUBLIC cublasZscal
cublasZscal PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5560]
    jmp rax
cublasZscal ENDP

PUBLIC cublasZscal_v2
cublasZscal_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5568]
    jmp rax
cublasZscal_v2 ENDP

PUBLIC cublasZscal_v2_64
cublasZscal_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5576]
    jmp rax
cublasZscal_v2_64 ENDP

PUBLIC cublasZswap
cublasZswap PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5584]
    jmp rax
cublasZswap ENDP

PUBLIC cublasZswap_v2
cublasZswap_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5592]
    jmp rax
cublasZswap_v2 ENDP

PUBLIC cublasZswap_v2_64
cublasZswap_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5600]
    jmp rax
cublasZswap_v2_64 ENDP

PUBLIC cublasZsymm
cublasZsymm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5608]
    jmp rax
cublasZsymm ENDP

PUBLIC cublasZsymm_v2
cublasZsymm_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5616]
    jmp rax
cublasZsymm_v2 ENDP

PUBLIC cublasZsymm_v2_64
cublasZsymm_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5624]
    jmp rax
cublasZsymm_v2_64 ENDP

PUBLIC cublasZsymv_v2
cublasZsymv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5632]
    jmp rax
cublasZsymv_v2 ENDP

PUBLIC cublasZsymv_v2_64
cublasZsymv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5640]
    jmp rax
cublasZsymv_v2_64 ENDP

PUBLIC cublasZsyr2_v2
cublasZsyr2_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5648]
    jmp rax
cublasZsyr2_v2 ENDP

PUBLIC cublasZsyr2_v2_64
cublasZsyr2_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5656]
    jmp rax
cublasZsyr2_v2_64 ENDP

PUBLIC cublasZsyr2k
cublasZsyr2k PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5664]
    jmp rax
cublasZsyr2k ENDP

PUBLIC cublasZsyr2k_v2
cublasZsyr2k_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5672]
    jmp rax
cublasZsyr2k_v2 ENDP

PUBLIC cublasZsyr2k_v2_64
cublasZsyr2k_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5680]
    jmp rax
cublasZsyr2k_v2_64 ENDP

PUBLIC cublasZsyr_v2
cublasZsyr_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5688]
    jmp rax
cublasZsyr_v2 ENDP

PUBLIC cublasZsyr_v2_64
cublasZsyr_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5696]
    jmp rax
cublasZsyr_v2_64 ENDP

PUBLIC cublasZsyrk
cublasZsyrk PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5704]
    jmp rax
cublasZsyrk ENDP

PUBLIC cublasZsyrk_v2
cublasZsyrk_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5712]
    jmp rax
cublasZsyrk_v2 ENDP

PUBLIC cublasZsyrk_v2_64
cublasZsyrk_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5720]
    jmp rax
cublasZsyrk_v2_64 ENDP

PUBLIC cublasZsyrkx
cublasZsyrkx PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5728]
    jmp rax
cublasZsyrkx ENDP

PUBLIC cublasZsyrkx_64
cublasZsyrkx_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5736]
    jmp rax
cublasZsyrkx_64 ENDP

PUBLIC cublasZtbmv
cublasZtbmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5744]
    jmp rax
cublasZtbmv ENDP

PUBLIC cublasZtbmv_v2
cublasZtbmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5752]
    jmp rax
cublasZtbmv_v2 ENDP

PUBLIC cublasZtbmv_v2_64
cublasZtbmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5760]
    jmp rax
cublasZtbmv_v2_64 ENDP

PUBLIC cublasZtbsv
cublasZtbsv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5768]
    jmp rax
cublasZtbsv ENDP

PUBLIC cublasZtbsv_v2
cublasZtbsv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5776]
    jmp rax
cublasZtbsv_v2 ENDP

PUBLIC cublasZtbsv_v2_64
cublasZtbsv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5784]
    jmp rax
cublasZtbsv_v2_64 ENDP

PUBLIC cublasZtpmv
cublasZtpmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5792]
    jmp rax
cublasZtpmv ENDP

PUBLIC cublasZtpmv_v2
cublasZtpmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5800]
    jmp rax
cublasZtpmv_v2 ENDP

PUBLIC cublasZtpmv_v2_64
cublasZtpmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5808]
    jmp rax
cublasZtpmv_v2_64 ENDP

PUBLIC cublasZtpsv
cublasZtpsv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5816]
    jmp rax
cublasZtpsv ENDP

PUBLIC cublasZtpsv_v2
cublasZtpsv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5824]
    jmp rax
cublasZtpsv_v2 ENDP

PUBLIC cublasZtpsv_v2_64
cublasZtpsv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5832]
    jmp rax
cublasZtpsv_v2_64 ENDP

PUBLIC cublasZtpttr
cublasZtpttr PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5840]
    jmp rax
cublasZtpttr ENDP

PUBLIC cublasZtrmm
cublasZtrmm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5848]
    jmp rax
cublasZtrmm ENDP

PUBLIC cublasZtrmm_v2
cublasZtrmm_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5856]
    jmp rax
cublasZtrmm_v2 ENDP

PUBLIC cublasZtrmm_v2_64
cublasZtrmm_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5864]
    jmp rax
cublasZtrmm_v2_64 ENDP

PUBLIC cublasZtrmv
cublasZtrmv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5872]
    jmp rax
cublasZtrmv ENDP

PUBLIC cublasZtrmv_v2
cublasZtrmv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5880]
    jmp rax
cublasZtrmv_v2 ENDP

PUBLIC cublasZtrmv_v2_64
cublasZtrmv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5888]
    jmp rax
cublasZtrmv_v2_64 ENDP

PUBLIC cublasZtrsm
cublasZtrsm PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5896]
    jmp rax
cublasZtrsm ENDP

PUBLIC cublasZtrsmBatched
cublasZtrsmBatched PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5904]
    jmp rax
cublasZtrsmBatched ENDP

PUBLIC cublasZtrsmBatched_64
cublasZtrsmBatched_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5912]
    jmp rax
cublasZtrsmBatched_64 ENDP

PUBLIC cublasZtrsm_v2
cublasZtrsm_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5920]
    jmp rax
cublasZtrsm_v2 ENDP

PUBLIC cublasZtrsm_v2_64
cublasZtrsm_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5928]
    jmp rax
cublasZtrsm_v2_64 ENDP

PUBLIC cublasZtrsv
cublasZtrsv PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5936]
    jmp rax
cublasZtrsv ENDP

PUBLIC cublasZtrsv_v2
cublasZtrsv_v2 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5944]
    jmp rax
cublasZtrsv_v2 ENDP

PUBLIC cublasZtrsv_v2_64
cublasZtrsv_v2_64 PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5952]
    jmp rax
cublasZtrsv_v2_64 ENDP

PUBLIC cublasZtrttp
cublasZtrttp PROC
    mov rax, QWORD PTR [g_real_cublas_funcs + 5960]
    jmp rax
cublasZtrttp ENDP

END