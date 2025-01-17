#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/elfnote-lto.h>
#include <linux/export-internal.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

#ifdef CONFIG_UNWINDER_ORC
#include <asm/orc_header.h>
ORC_HEADER;
#endif

BUILD_SALT;
BUILD_LTO_INFO;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

KSYMTAB_FUNC(nvKmsKapiGetFunctionsTable, "", "");

SYMBOL_CRC(nvKmsKapiGetFunctionsTable, 0x19e7f00b, "");

static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0x67543840, "filp_open" },
	{ 0xc31db0ce, "is_vmalloc_addr" },
	{ 0x88db9f48, "__check_object_size" },
	{ 0x9e7d6bd0, "__udelay" },
	{ 0x7b4da6ff, "__init_rwsem" },
	{ 0xa4b3adb, "backlight_device_unregister" },
	{ 0x13c49cc2, "_copy_from_user" },
	{ 0xafd744c6, "__x86_indirect_thunk_rbp" },
	{ 0x8db648c7, "vmalloc_to_page" },
	{ 0xb0e602eb, "memmove" },
	{ 0x656e4a6e, "snprintf" },
	{ 0xa6257a2f, "complete" },
	{ 0x4c236f6f, "__x86_indirect_thunk_r15" },
	{ 0x26049793, "fget" },
	{ 0xcf2a6966, "up" },
	{ 0x2b2c878a, "proc_mkdir_mode" },
	{ 0x69acdf38, "memcpy" },
	{ 0x37a0cba, "kfree" },
	{ 0xb6a5193f, "pcpu_hot" },
	{ 0xa389e45a, "seq_lseek" },
	{ 0x46232bfb, "proc_create_data" },
	{ 0xb3f7646e, "kthread_should_stop" },
	{ 0x82ee90dc, "timer_delete_sync" },
	{ 0xe2964344, "__wake_up" },
	{ 0x148653, "vsnprintf" },
	{ 0x34db050b, "_raw_spin_lock_irqsave" },
	{ 0xc3ff38c2, "down_read_trylock" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0xbca77c8c, "wake_up_process" },
	{ 0x4482cdb, "__refrigerator" },
	{ 0x284faa6b, "__x86_indirect_thunk_r11" },
	{ 0x65487097, "__x86_indirect_thunk_rax" },
	{ 0xd73653c4, "freezer_active" },
	{ 0x6b2dc060, "dump_stack" },
	{ 0x122c3a7e, "_printk" },
	{ 0x193c3c08, "nvidia_get_rm_ops" },
	{ 0x1000e51, "schedule" },
	{ 0xf0fdf6cb, "__stack_chk_fail" },
	{ 0x6383b27c, "__x86_indirect_thunk_rdx" },
	{ 0x87a21cb3, "__ubsan_handle_out_of_bounds" },
	{ 0xf4407d6b, "cdev_add" },
	{ 0x907caf15, "fput" },
	{ 0x57bc19d2, "down_write" },
	{ 0xce807a25, "up_write" },
	{ 0x55385e2e, "__x86_indirect_thunk_r14" },
	{ 0xc38c83b8, "mod_timer" },
	{ 0x6626afca, "down" },
	{ 0x670ecece, "__x86_indirect_thunk_rbx" },
	{ 0x9166fada, "strncpy" },
	{ 0x1a79c8e9, "__x86_indirect_thunk_r13" },
	{ 0x1edb69d6, "ktime_get_raw_ts64" },
	{ 0x449ad0a7, "memcmp" },
	{ 0x908dcd2e, "kthread_stop" },
	{ 0x55b76df3, "freezing_slow_path" },
	{ 0xd35cce70, "_raw_spin_unlock_irqrestore" },
	{ 0xfb578fc5, "memset" },
	{ 0x31549b2a, "__x86_indirect_thunk_r10" },
	{ 0x7d04ae5a, "param_ops_charp" },
	{ 0x92f3de2f, "kernel_read" },
	{ 0x25974000, "wait_for_completion" },
	{ 0x5b8239ca, "__x86_return_thunk" },
	{ 0x6b10bee1, "_copy_to_user" },
	{ 0xd9a5ea54, "__init_waitqueue_head" },
	{ 0xce168946, "proc_remove" },
	{ 0xe2d5255a, "strcmp" },
	{ 0x668b19a1, "down_read" },
	{ 0x15ba50a6, "jiffies" },
	{ 0x5d626f20, "kthread_create_on_node" },
	{ 0xe85f2892, "seq_read" },
	{ 0x999e8297, "vfree" },
	{ 0x6091b333, "unregister_chrdev_region" },
	{ 0xc6f46339, "init_timer_key" },
	{ 0x1959c9a1, "param_ops_bool" },
	{ 0x3ef70737, "filp_close" },
	{ 0x66cca4f9, "__x86_indirect_thunk_rcx" },
	{ 0x56470118, "__warn_printk" },
	{ 0x6bd0e573, "down_interruptible" },
	{ 0xe0112fc4, "__x86_indirect_thunk_r9" },
	{ 0x6a5198d7, "backlight_device_register" },
	{ 0x7de7bf50, "__acpi_video_get_backlight_type" },
	{ 0xc07351b3, "__SCT__cond_resched" },
	{ 0x834ea9d, "seq_puts" },
	{ 0x981b202c, "single_release" },
	{ 0x362f9a8, "__x86_indirect_thunk_r12" },
	{ 0x4454730e, "kmalloc_trace" },
	{ 0x3fd78f3b, "register_chrdev_region" },
	{ 0x54b1fac6, "__ubsan_handle_load_invalid_value" },
	{ 0x754d539c, "strlen" },
	{ 0x3a0a8b87, "param_ops_int" },
	{ 0xff0f94f9, "single_open" },
	{ 0xd6ee688f, "vmalloc" },
	{ 0x53b954a2, "up_read" },
	{ 0xf90a1e85, "__x86_indirect_thunk_r8" },
	{ 0xf9a482f9, "msleep" },
	{ 0xa6f7a612, "cdev_init" },
	{ 0xeb233a45, "__kmalloc" },
	{ 0xe2c17b5d, "__SCT__might_resched" },
	{ 0xb88db70c, "kmalloc_caches" },
	{ 0x8f44466e, "cdev_del" },
	{ 0x2fa5cadd, "module_layout" },
};

MODULE_INFO(depends, "nvidia,video");


MODULE_INFO(srcversion, "852A75E1965758498127F16");
