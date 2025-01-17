savedcmd_/home/tallen93/dev/uvm-eviction/drivers/x86_64-555.42.02/faults/kernel-open/nvidia-uvm/nvstatus.o := cc -Wp,-MMD,/home/tallen93/dev/uvm-eviction/drivers/x86_64-555.42.02/faults/kernel-open/nvidia-uvm/.nvstatus.o.d -nostdinc -I./arch/x86/include -I./arch/x86/include/generated  -I./include -I./arch/x86/include/uapi -I./arch/x86/include/generated/uapi -I./include/uapi -I./include/generated/uapi -include ./include/linux/compiler-version.h -include ./include/linux/kconfig.h -I./ubuntu/include -include ./include/linux/compiler_types.h -D__KERNEL__ -fmacro-prefix-map=./= -std=gnu11 -fshort-wchar -funsigned-char -fno-common -fno-PIE -fno-strict-aliasing -Wall -Wundef -Werror=implicit-function-declaration -Werror=implicit-int -Werror=return-type -Werror=strict-prototypes -Wno-format-security -Wno-trigraphs -mno-sse -mno-mmx -mno-sse2 -mno-3dnow -mno-avx -fcf-protection=none -m64 -falign-jumps=1 -falign-loops=1 -mno-80387 -mno-fp-ret-in-387 -mpreferred-stack-boundary=3 -mskip-rax-setup -mtune=generic -mno-red-zone -mcmodel=kernel -Wno-sign-compare -fno-asynchronous-unwind-tables -mindirect-branch=thunk-extern -mindirect-branch-register -mindirect-branch-cs-prefix -mfunction-return=thunk-extern -fno-jump-tables -mharden-sls=all -fpatchable-function-entry=16,16 -fno-delete-null-pointer-checks -Wno-frame-address -Wno-format-truncation -Wno-format-overflow -Wno-address-of-packed-member -O2 -fno-allow-store-data-races -Wframe-larger-than=1024 -fstack-protector-strong -Wno-main -Wno-unused-but-set-variable -Wno-unused-const-variable -Wno-dangling-pointer -fno-omit-frame-pointer -fno-optimize-sibling-calls -ftrivial-auto-var-init=zero -fno-stack-clash-protection -fzero-call-used-regs=used-gpr -pg -mrecord-mcount -mfentry -DCC_USING_FENTRY -falign-functions=16 -Wvla -Wno-pointer-sign -Wcast-function-type -Wno-stringop-truncation -Wno-stringop-overflow -Wno-restrict -Wno-maybe-uninitialized -Wno-array-bounds -Wno-alloc-size-larger-than -Wimplicit-fallthrough=5 -fno-strict-overflow -fno-stack-check -fconserve-stack -Werror=date-time -Werror=incompatible-pointer-types -Werror=designated-init -Wno-packed-not-aligned -g -gdwarf-5 -Wno-error -I/home/tallen93/dev/uvm-eviction/drivers/x86_64-555.42.02/faults/kernel-open/common/inc -I/home/tallen93/dev/uvm-eviction/drivers/x86_64-555.42.02/faults/kernel-open -Wall -Wno-cast-qual -Wno-format-extra-args -D__KERNEL__ -DMODULE -DNVRM -DNV_VERSION_STRING=\"555.42.02\" -DNV_FILESYSTEM_ACCESS_AVAILABLE=1 -Wno-unused-function -Wuninitialized -fno-strict-aliasing -ffreestanding -mno-red-zone -mcmodel=kernel -DNV_UVM_ENABLE -Werror=undef -DNV_SPECTRE_V2=0 -DNV_KERNEL_INTERFACE_LAYER -O2 -DNVIDIA_UVM_ENABLED -DNVIDIA_UNDEF_LEGACY_BIT_MACROS -DLinux -D__linux__ -I/home/tallen93/dev/uvm-eviction/drivers/x86_64-555.42.02/faults/kernel-open/nvidia-uvm  -fsanitize=bounds-strict -fsanitize=shift -fsanitize=bool -fsanitize=enum  -DMODULE  -DKBUILD_BASENAME='"nvstatus"' -DKBUILD_MODNAME='"nvidia_uvm"' -D__KBUILD_MODNAME=kmod_nvidia_uvm -c -o /home/tallen93/dev/uvm-eviction/drivers/x86_64-555.42.02/faults/kernel-open/nvidia-uvm/nvstatus.o /home/tallen93/dev/uvm-eviction/drivers/x86_64-555.42.02/faults/kernel-open/nvidia-uvm/nvstatus.c   ; ./tools/objtool/objtool --hacks=jump_label --hacks=noinstr --hacks=skylake --retpoline --rethunk --sls --stackval --static-call --uaccess --prefix=16   --module /home/tallen93/dev/uvm-eviction/drivers/x86_64-555.42.02/faults/kernel-open/nvidia-uvm/nvstatus.o

source_/home/tallen93/dev/uvm-eviction/drivers/x86_64-555.42.02/faults/kernel-open/nvidia-uvm/nvstatus.o := /home/tallen93/dev/uvm-eviction/drivers/x86_64-555.42.02/faults/kernel-open/nvidia-uvm/nvstatus.c

deps_/home/tallen93/dev/uvm-eviction/drivers/x86_64-555.42.02/faults/kernel-open/nvidia-uvm/nvstatus.o := \
  include/linux/compiler-version.h \
    $(wildcard include/config/CC_VERSION_TEXT) \
  include/linux/kconfig.h \
    $(wildcard include/config/CPU_BIG_ENDIAN) \
    $(wildcard include/config/BOOGER) \
    $(wildcard include/config/FOO) \
  include/linux/compiler_types.h \
    $(wildcard include/config/DEBUG_INFO_BTF) \
    $(wildcard include/config/PAHOLE_HAS_BTF_TAG) \
    $(wildcard include/config/FUNCTION_ALIGNMENT) \
    $(wildcard include/config/CC_IS_GCC) \
    $(wildcard include/config/HAVE_ARCH_COMPILER_H) \
    $(wildcard include/config/CC_HAS_ASM_INLINE) \
  include/linux/compiler_attributes.h \
  include/linux/compiler-gcc.h \
    $(wildcard include/config/RETPOLINE) \
    $(wildcard include/config/ARCH_USE_BUILTIN_BSWAP) \
    $(wildcard include/config/SHADOW_CALL_STACK) \
    $(wildcard include/config/KCOV) \
  /home/tallen93/dev/uvm-eviction/drivers/x86_64-555.42.02/faults/kernel-open/common/inc/nvstatus.h \
  /home/tallen93/dev/uvm-eviction/drivers/x86_64-555.42.02/faults/kernel-open/common/inc/nvtypes.h \
  /home/tallen93/dev/uvm-eviction/drivers/x86_64-555.42.02/faults/kernel-open/common/inc/cpuopsys.h \
  /home/tallen93/dev/uvm-eviction/drivers/x86_64-555.42.02/faults/kernel-open/common/inc/nvstatuscodes.h \

/home/tallen93/dev/uvm-eviction/drivers/x86_64-555.42.02/faults/kernel-open/nvidia-uvm/nvstatus.o: $(deps_/home/tallen93/dev/uvm-eviction/drivers/x86_64-555.42.02/faults/kernel-open/nvidia-uvm/nvstatus.o)

$(deps_/home/tallen93/dev/uvm-eviction/drivers/x86_64-555.42.02/faults/kernel-open/nvidia-uvm/nvstatus.o):

/home/tallen93/dev/uvm-eviction/drivers/x86_64-555.42.02/faults/kernel-open/nvidia-uvm/nvstatus.o: $(wildcard ./tools/objtool/objtool)
