	.file	"<string>"
	.section	.ltext,"axl",@progbits
	.globl	_ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy
	.p2align	4
	.type	_ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy,@function
_ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy:
	tzcntq	%rdx, %rax
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
	je	.LBB1_6
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
.LBB1_9:
	movabsq	$_ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy, %rax
	leaq	8(%rsp), %rdi
	movq	%rbx, %rdx
	callq	*%rax
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
.LBB1_4:
	.cfi_def_cfa_offset 48
	movabsq	$PyExc_RuntimeError, %rdi
	movabsq	$".const.missing Environment: _ZN08NumbaEnv8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy", %rsi
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
.LBB1_6:
	.cfi_def_cfa_offset 48
	xorl	%ebx, %ebx
	movabsq	$PyErr_Occurred, %rax
	callq	*%rax
	testq	%rax, %rax
	je	.LBB1_9
	jmp	.LBB1_1
.Lfunc_end1:
	.size	_ZN7cpython8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy, .Lfunc_end1-_ZN7cpython8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy
	.cfi_endproc

	.globl	cfunc._ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy
	.p2align	4
	.type	cfunc._ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy,@function
cfunc._ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy:
	pushq	%rax
	movq	%rdi, %rdx
	movabsq	$_ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy, %rax
	movq	%rsp, %rdi
	callq	*%rax
	movq	(%rsp), %rax
	popq	%rcx
	retq
.Lfunc_end2:
	.size	cfunc._ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy, .Lfunc_end2-cfunc._ZN8bitboard3lsbB2v2B58c8tJTIeFIjxB2IKSgI4CrvQClYa4yxbNYYk55YmVxeqawBAsgqjUBAA_3dEy

	.type	.const.lsb,@object
	.section	.lrodata,"al",@progbits
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

	.section	".note.GNU-stack","",@progbits
