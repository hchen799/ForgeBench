; ModuleID = '/export/hdd/scratch/hchen799/hls_benchmark/merged_conv/compile/project_1/solution1/.autopilot/db/a.g.ld.5.gdce.bc'
source_filename = "llvm-link"
target datalayout = "e-m:e-i64:64-i128:128-i256:256-i512:512-i1024:1024-i2048:2048-i4096:4096-n8:16:32:64-S128-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "fpga64-xilinx-none"

; Function Attrs: noinline
define void @apatb_conv_via_tiling_ir(i32 %in_ch, i32 %out_ch, i32 %H, i32 %W, [256 x float]* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="256" "maxi" %input, [256 x [3 x [3 x float]]]* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="256" "maxi" %weight, float* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="256" "maxi" %bias, [256 x float]* noalias nocapture nonnull "fpga.decayed.dim.hint"="256" "maxi" %output, i32 %stride, i32 %pad) local_unnamed_addr #0 {
entry:
  %malloccall = tail call i8* @malloc(i64 262144)
  %input_copy = bitcast i8* %malloccall to [256 x [256 x float]]*
  %malloccall1 = tail call i8* @malloc(i64 2359296)
  %weight_copy = bitcast i8* %malloccall1 to [256 x [256 x [3 x [3 x float]]]]*
  %bias_copy = alloca [256 x float], align 512
  %malloccall2 = tail call i8* @malloc(i64 262144)
  %output_copy = bitcast i8* %malloccall2 to [256 x [256 x float]]*
  %0 = bitcast [256 x float]* %input to [256 x [256 x float]]*
  %1 = bitcast [256 x [3 x [3 x float]]]* %weight to [256 x [256 x [3 x [3 x float]]]]*
  %2 = bitcast float* %bias to [256 x float]*
  %3 = bitcast [256 x float]* %output to [256 x [256 x float]]*
  call fastcc void @copy_in([256 x [256 x float]]* nonnull %0, [256 x [256 x float]]* %input_copy, [256 x [256 x [3 x [3 x float]]]]* nonnull %1, [256 x [256 x [3 x [3 x float]]]]* %weight_copy, [256 x float]* nonnull %2, [256 x float]* nonnull align 512 %bias_copy, [256 x [256 x float]]* nonnull %3, [256 x [256 x float]]* %output_copy)
  call void @apatb_conv_via_tiling_hw(i32 %in_ch, i32 %out_ch, i32 %H, i32 %W, [256 x [256 x float]]* %input_copy, [256 x [256 x [3 x [3 x float]]]]* %weight_copy, [256 x float]* %bias_copy, [256 x [256 x float]]* %output_copy, i32 %stride, i32 %pad)
  call void @copy_back([256 x [256 x float]]* %0, [256 x [256 x float]]* %input_copy, [256 x [256 x [3 x [3 x float]]]]* %1, [256 x [256 x [3 x [3 x float]]]]* %weight_copy, [256 x float]* %2, [256 x float]* %bias_copy, [256 x [256 x float]]* %3, [256 x [256 x float]]* %output_copy)
  tail call void @free(i8* %malloccall)
  tail call void @free(i8* %malloccall1)
  tail call void @free(i8* %malloccall2)
  ret void
}

declare noalias i8* @malloc(i64) local_unnamed_addr

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @copy_in([256 x [256 x float]]* noalias readonly, [256 x [256 x float]]* noalias, [256 x [256 x [3 x [3 x float]]]]* noalias readonly, [256 x [256 x [3 x [3 x float]]]]* noalias, [256 x float]* noalias readonly, [256 x float]* noalias align 512, [256 x [256 x float]]* noalias readonly, [256 x [256 x float]]* noalias) unnamed_addr #1 {
entry:
  call fastcc void @onebyonecpy_hls.p0a256a256f32([256 x [256 x float]]* %1, [256 x [256 x float]]* %0)
  call fastcc void @onebyonecpy_hls.p0a256a256a3a3f32([256 x [256 x [3 x [3 x float]]]]* %3, [256 x [256 x [3 x [3 x float]]]]* %2)
  call fastcc void @onebyonecpy_hls.p0a256f32([256 x float]* align 512 %5, [256 x float]* %4)
  call fastcc void @onebyonecpy_hls.p0a256a256f32([256 x [256 x float]]* %7, [256 x [256 x float]]* %6)
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @onebyonecpy_hls.p0a256a256f32([256 x [256 x float]]* noalias %dst, [256 x [256 x float]]* noalias readonly %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [256 x [256 x float]]* %dst, null
  %1 = icmp eq [256 x [256 x float]]* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @arraycpy_hls.p0a256a256f32([256 x [256 x float]]* nonnull %dst, [256 x [256 x float]]* nonnull %src, i64 256)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a256a256f32([256 x [256 x float]]* %dst, [256 x [256 x float]]* readonly %src, i64 %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [256 x [256 x float]]* %src, null
  %1 = icmp eq [256 x [256 x float]]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [256 x [256 x float]], [256 x [256 x float]]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [256 x [256 x float]], [256 x [256 x float]]* %src, i64 0, i64 %for.loop.idx2
  call void @arraycpy_hls.p0a256f32([256 x float]* %dst.addr, [256 x float]* %src.addr, i64 256)
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a256f32([256 x float]* %dst, [256 x float]* readonly %src, i64 %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [256 x float]* %src, null
  %1 = icmp eq [256 x float]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [256 x float], [256 x float]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [256 x float], [256 x float]* %src, i64 0, i64 %for.loop.idx2
  %3 = load float, float* %src.addr, align 4
  store float %3, float* %dst.addr, align 4
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @onebyonecpy_hls.p0a256a256a3a3f32([256 x [256 x [3 x [3 x float]]]]* noalias %dst, [256 x [256 x [3 x [3 x float]]]]* noalias readonly %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [256 x [256 x [3 x [3 x float]]]]* %dst, null
  %1 = icmp eq [256 x [256 x [3 x [3 x float]]]]* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @arraycpy_hls.p0a256a256a3a3f32([256 x [256 x [3 x [3 x float]]]]* nonnull %dst, [256 x [256 x [3 x [3 x float]]]]* nonnull %src, i64 256)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a256a256a3a3f32([256 x [256 x [3 x [3 x float]]]]* %dst, [256 x [256 x [3 x [3 x float]]]]* readonly %src, i64 %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [256 x [256 x [3 x [3 x float]]]]* %src, null
  %1 = icmp eq [256 x [256 x [3 x [3 x float]]]]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [256 x [256 x [3 x [3 x float]]]], [256 x [256 x [3 x [3 x float]]]]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [256 x [256 x [3 x [3 x float]]]], [256 x [256 x [3 x [3 x float]]]]* %src, i64 0, i64 %for.loop.idx2
  call void @arraycpy_hls.p0a256a3a3f32([256 x [3 x [3 x float]]]* %dst.addr, [256 x [3 x [3 x float]]]* %src.addr, i64 256)
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a256a3a3f32([256 x [3 x [3 x float]]]* %dst, [256 x [3 x [3 x float]]]* readonly %src, i64 %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [256 x [3 x [3 x float]]]* %src, null
  %1 = icmp eq [256 x [3 x [3 x float]]]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [256 x [3 x [3 x float]]], [256 x [3 x [3 x float]]]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [256 x [3 x [3 x float]]], [256 x [3 x [3 x float]]]* %src, i64 0, i64 %for.loop.idx2
  call void @arraycpy_hls.p0a3a3f32([3 x [3 x float]]* %dst.addr, [3 x [3 x float]]* %src.addr, i64 3)
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a3a3f32([3 x [3 x float]]* %dst, [3 x [3 x float]]* readonly %src, i64 %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [3 x [3 x float]]* %src, null
  %1 = icmp eq [3 x [3 x float]]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [3 x [3 x float]], [3 x [3 x float]]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [3 x [3 x float]], [3 x [3 x float]]* %src, i64 0, i64 %for.loop.idx2
  call void @arraycpy_hls.p0a3f32([3 x float]* %dst.addr, [3 x float]* %src.addr, i64 3)
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a3f32([3 x float]* %dst, [3 x float]* readonly %src, i64 %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [3 x float]* %src, null
  %1 = icmp eq [3 x float]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [3 x float], [3 x float]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [3 x float], [3 x float]* %src, i64 0, i64 %for.loop.idx2
  %3 = load float, float* %src.addr, align 4
  store float %3, float* %dst.addr, align 4
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @onebyonecpy_hls.p0a256f32([256 x float]* noalias align 512 %dst, [256 x float]* noalias readonly %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [256 x float]* %dst, null
  %1 = icmp eq [256 x float]* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @arraycpy_hls.p0a256f32([256 x float]* nonnull %dst, [256 x float]* nonnull %src, i64 256)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @copy_out([256 x [256 x float]]* noalias, [256 x [256 x float]]* noalias readonly, [256 x [256 x [3 x [3 x float]]]]* noalias, [256 x [256 x [3 x [3 x float]]]]* noalias readonly, [256 x float]* noalias, [256 x float]* noalias readonly align 512, [256 x [256 x float]]* noalias, [256 x [256 x float]]* noalias readonly) unnamed_addr #4 {
entry:
  call fastcc void @onebyonecpy_hls.p0a256a256f32([256 x [256 x float]]* %0, [256 x [256 x float]]* %1)
  call fastcc void @onebyonecpy_hls.p0a256a256a3a3f32([256 x [256 x [3 x [3 x float]]]]* %2, [256 x [256 x [3 x [3 x float]]]]* %3)
  call fastcc void @onebyonecpy_hls.p0a256f32([256 x float]* %4, [256 x float]* align 512 %5)
  call fastcc void @onebyonecpy_hls.p0a256a256f32([256 x [256 x float]]* %6, [256 x [256 x float]]* %7)
  ret void
}

declare void @free(i8*) local_unnamed_addr

declare void @apatb_conv_via_tiling_hw(i32, i32, i32, i32, [256 x [256 x float]]*, [256 x [256 x [3 x [3 x float]]]]*, [256 x float]*, [256 x [256 x float]]*, i32, i32)

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @copy_back([256 x [256 x float]]* noalias, [256 x [256 x float]]* noalias readonly, [256 x [256 x [3 x [3 x float]]]]* noalias, [256 x [256 x [3 x [3 x float]]]]* noalias readonly, [256 x float]* noalias, [256 x float]* noalias readonly align 512, [256 x [256 x float]]* noalias, [256 x [256 x float]]* noalias readonly) unnamed_addr #4 {
entry:
  call fastcc void @onebyonecpy_hls.p0a256a256f32([256 x [256 x float]]* %6, [256 x [256 x float]]* %7)
  ret void
}

define void @conv_via_tiling_hw_stub_wrapper(i32, i32, i32, i32, [256 x [256 x float]]*, [256 x [256 x [3 x [3 x float]]]]*, [256 x float]*, [256 x [256 x float]]*, i32, i32) #5 {
entry:
  call void @copy_out([256 x [256 x float]]* null, [256 x [256 x float]]* %4, [256 x [256 x [3 x [3 x float]]]]* null, [256 x [256 x [3 x [3 x float]]]]* %5, [256 x float]* null, [256 x float]* %6, [256 x [256 x float]]* null, [256 x [256 x float]]* %7)
  %10 = bitcast [256 x [256 x float]]* %4 to [256 x float]*
  %11 = bitcast [256 x [256 x [3 x [3 x float]]]]* %5 to [256 x [3 x [3 x float]]]*
  %12 = bitcast [256 x float]* %6 to float*
  %13 = bitcast [256 x [256 x float]]* %7 to [256 x float]*
  call void @conv_via_tiling_hw_stub(i32 %0, i32 %1, i32 %2, i32 %3, [256 x float]* %10, [256 x [3 x [3 x float]]]* %11, float* %12, [256 x float]* %13, i32 %8, i32 %9)
  call void @copy_in([256 x [256 x float]]* null, [256 x [256 x float]]* %4, [256 x [256 x [3 x [3 x float]]]]* null, [256 x [256 x [3 x [3 x float]]]]* %5, [256 x float]* null, [256 x float]* %6, [256 x [256 x float]]* null, [256 x [256 x float]]* %7)
  ret void
}

declare void @conv_via_tiling_hw_stub(i32, i32, i32, i32, [256 x float]* noalias nocapture nonnull readonly, [256 x [3 x [3 x float]]]* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, [256 x float]* noalias nocapture nonnull, i32, i32)

attributes #0 = { noinline "fpga.wrapper.func"="wrapper" }
attributes #1 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="copyin" }
attributes #2 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="onebyonecpy_hls" }
attributes #3 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="arraycpy_hls" }
attributes #4 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="copyout" }
attributes #5 = { "fpga.wrapper.func"="stub" }

!llvm.dbg.cu = !{}
!llvm.ident = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!llvm.module.flags = !{!1, !2, !3}
!blackbox_cfg = !{!4}

!0 = !{!"clang version 7.0.0 "}
!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{}
