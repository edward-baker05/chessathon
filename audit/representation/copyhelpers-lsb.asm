	.file	"<string>"
	.section	.ltext,"axl",@progbits
	.globl	_ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy
	.p2align	4
	.type	_ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy,@function
_ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy:
	blsiq	%rdx, %rax
	movabsq	$285870213051386505, %rcx
	imulq	%rax, %rcx
	movabsq	$.const.array.data, %rax
	shrq	$58, %rcx
	movq	(%rax,%rcx,8), %rax
	movq	%rax, (%rdi)
	xorl	%eax, %eax
	retq
.Lfunc_end0:
	.size	_ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy, .Lfunc_end0-_ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy

	.globl	_ZN7cpython8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy
	.p2align	4
	.type	_ZN7cpython8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy,@function
_ZN7cpython8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy:
	.cfi_startproc
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	subq	$24, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	movq	%rsi, %rdi
	movabsq	$.const.lsb, %rsi
	movabsq	$PyArg_UnpackTuple, %r9
	leaq	16(%rsp), %r8
	movl	$1, %edx
	movl	$1, %ecx
	xorl	%eax, %eax
	callq	*%r9
	testl	%eax, %eax
	je	.LBB1_1
	movabsq	$_ZN08NumbaEnv8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy, %rax
	cmpq	$0, (%rax)
	je	.LBB1_4
	movq	16(%rsp), %rdi
	movabsq	$PyNumber_Long, %rax
	callq	*%rax
	testq	%rax, %rax
	je	.LBB1_7
	movq	%rax, %r14
	movabsq	$PyLong_AsUnsignedLongLong, %rax
	movq	%r14, %rdi
	callq	*%rax
	movq	%rax, %rbx
	movabsq	$Py_DecRef, %rax
	movq	%r14, %rdi
	callq	*%rax
	movabsq	$PyErr_Occurred, %rax
	callq	*%rax
	testq	%rax, %rax
	jne	.LBB1_1
.LBB1_10:
	movabsq	$_ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy, %rax
	leaq	8(%rsp), %rdi
	movq	%rbx, %rdx
	callq	*%rax
	testl	%eax, %eax
	jne	.LBB1_11
	movq	8(%rsp), %rdi
	movabsq	$PyLong_FromLongLong, %rax
	callq	*%rax
	addq	$24, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.LBB1_11:
	.cfi_def_cfa_offset 48
	jg	.LBB1_17
	cmpl	$-1, %eax
	je	.LBB1_1
	cmpl	$-3, %eax
	jne	.LBB1_15
	movabsq	$PyExc_StopIteration, %rdi
	movabsq	$PyErr_SetNone, %rax
	callq	*%rax
	jmp	.LBB1_1
.LBB1_4:
	movabsq	$PyExc_RuntimeError, %rdi
	movabsq	$".const.missing Environment: _ZN08NumbaEnv8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy", %rsi
	jmp	.LBB1_5
.LBB1_7:
	xorl	%ebx, %ebx
	movabsq	$PyErr_Occurred, %rax
	callq	*%rax
	testq	%rax, %rax
	je	.LBB1_10
	jmp	.LBB1_1
.LBB1_15:
	movabsq	$PyExc_SystemError, %rdi
	movabsq	$".const.unknown error when calling native function", %rsi
.LBB1_5:
	movabsq	$PyErr_SetString, %rax
	callq	*%rax
.LBB1_1:
	xorl	%eax, %eax
	addq	$24, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.LBB1_17:
	.cfi_def_cfa_offset 48
	movabsq	$PyErr_Clear, %rax
	callq	*%rax
.Lfunc_end1:
	.size	_ZN7cpython8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy, .Lfunc_end1-_ZN7cpython8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy
	.cfi_endproc

	.globl	cfunc._ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy
	.p2align	4
	.type	cfunc._ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy,@function
cfunc._ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy:
	.cfi_startproc
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	subq	$24, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	movq	%rdi, %rdx
	movabsq	$_ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy, %rax
	leaq	16(%rsp), %rdi
	callq	*%rax
	movq	16(%rsp), %rcx
	movl	$0, 12(%rsp)
	testl	%eax, %eax
	jne	.LBB2_1
.LBB2_5:
	movq	%rcx, %rax
	addq	$24, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.LBB2_1:
	.cfi_def_cfa_offset 48
	movq	%rcx, %r14
	movabsq	$numba_gil_ensure, %rcx
	leaq	12(%rsp), %rdi
	movl	%eax, %ebx
	callq	*%rcx
	testl	%ebx, %ebx
	jg	.LBB2_2
	movl	%ebx, %eax
	cmpl	$-1, %ebx
	je	.LBB2_4
	cmpl	$-3, %eax
	jne	.LBB2_3
	movabsq	$PyExc_StopIteration, %rdi
	movabsq	$PyErr_SetNone, %rax
	callq	*%rax
	jmp	.LBB2_4
.LBB2_3:
	movabsq	$PyExc_SystemError, %rdi
	movabsq	$".const.unknown error when calling native function.2", %rsi
	movabsq	$PyErr_SetString, %rax
	callq	*%rax
.LBB2_4:
	movabsq	$".const.<numba.core.cpu.CPUContext>", %rdi
	movabsq	$PyUnicode_FromString, %rax
	callq	*%rax
	movq	%rax, %rbx
	movabsq	$PyErr_WriteUnraisable, %rax
	movq	%rbx, %rdi
	callq	*%rax
	movabsq	$Py_DecRef, %rax
	movq	%rbx, %rdi
	callq	*%rax
	movabsq	$numba_gil_release, %rax
	leaq	12(%rsp), %rdi
	callq	*%rax
	movq	%r14, %rcx
	jmp	.LBB2_5
.LBB2_2:
	movabsq	$PyErr_Clear, %rax
	callq	*%rax
.Lfunc_end2:
	.size	cfunc._ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy, .Lfunc_end2-cfunc._ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy
	.cfi_endproc

	.type	.const.array.data,@object
	.section	.lrodata,"al",@progbits
	.p2align	3, 0x0
.const.array.data:
	.asciz	"\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\0000\000\000\000\000\000\000\000\002\000\000\000\000\000\000\0009\000\000\000\000\000\000\0001\000\000\000\000\000\000\000\034\000\000\000\000\000\000\000\003\000\000\000\000\000\000\000=\000\000\000\000\000\000\000:\000\000\000\000\000\000\0002\000\000\000\000\000\000\000*\000\000\000\000\000\000\000&\000\000\000\000\000\000\000\035\000\000\000\000\000\000\000\021\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000>\000\000\000\000\000\000\0007\000\000\000\000\000\000\000;\000\000\000\000\000\000\000$\000\000\000\000\000\000\0005\000\000\000\000\000\000\0003\000\000\000\000\000\000\000+\000\000\000\000\000\000\000\026\000\000\000\000\000\000\000-\000\000\000\000\000\000\000'\000\000\000\000\000\000\000!\000\000\000\000\000\000\000\036\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\022\000\000\000\000\000\000\000\f\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000?\000\000\000\000\000\000\000/\000\000\000\000\000\000\0008\000\000\000\000\000\000\000\033\000\000\000\000\000\000\000<\000\000\000\000\000\000\000)\000\000\000\000\000\000\000%\000\000\000\000\000\000\000\020\000\000\000\000\000\000\0006\000\000\000\000\000\000\000#\000\000\000\000\000\000\0004\000\000\000\000\000\000\000\025\000\000\000\000\000\000\000,\000\000\000\000\000\000\000 \000\000\000\000\000\000\000\027\000\000\000\000\000\000\000\013\000\000\000\000\000\000\000.\000\000\000\000\000\000\000\032\000\000\000\000\000\000\000(\000\000\000\000\000\000\000\017\000\000\000\000\000\000\000\"\000\000\000\000\000\000\000\024\000\000\000\000\000\000\000\037\000\000\000\000\000\000\000\n\000\000\000\000\000\000\000\031\000\000\000\000\000\000\000\016\000\000\000\000\000\000\000\023\000\000\000\000\000\000\000\t\000\000\000\000\000\000\000\r\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\007\000\000\000\000\000\000\000\006\000\000\000\000\000\000"
	.size	.const.array.data, 512

	.type	.const.lsb,@object
.const.lsb:
	.asciz	"lsb"
	.size	.const.lsb, 4

	.type	_ZN08NumbaEnv8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy,@object
	.comm	_ZN08NumbaEnv8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy,8,8
	.type	".const.missing Environment: _ZN08NumbaEnv8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy",@object
	.p2align	4, 0x0
".const.missing Environment: _ZN08NumbaEnv8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy":
	.asciz	"missing Environment: _ZN08NumbaEnv8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy"
	.size	".const.missing Environment: _ZN08NumbaEnv8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy", 115

	.type	".const.unknown error when calling native function",@object
	.p2align	4, 0x0
".const.unknown error when calling native function":
	.asciz	"unknown error when calling native function"
	.size	".const.unknown error when calling native function", 43

	.type	".const.unknown error when calling native function.2",@object
	.p2align	4, 0x0
".const.unknown error when calling native function.2":
	.asciz	"unknown error when calling native function"
	.size	".const.unknown error when calling native function.2", 43

	.type	".const.<numba.core.cpu.CPUContext>",@object
	.p2align	4, 0x0
".const.<numba.core.cpu.CPUContext>":
	.asciz	"<numba.core.cpu.CPUContext>"
	.size	".const.<numba.core.cpu.CPUContext>", 28

	.type	_ZN08NumbaEnv5numba7cpython8builtins6ol_int12_3clocals_3e4implB2v3B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEy,@object
	.comm	_ZN08NumbaEnv5numba7cpython8builtins6ol_int12_3clocals_3e4implB2v3B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEy,8,8
	.section	".note.GNU-stack","",@progbits
