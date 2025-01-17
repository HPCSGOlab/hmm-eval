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



static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0xc8c85086, "sg_free_table" },
	{ 0x37a0cba, "kfree" },
	{ 0xb6a5193f, "pcpu_hot" },
	{ 0x6200d00d, "__module_get" },
	{ 0x161140df, "nvidia_p2p_dma_unmap_pages" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0x65487097, "__x86_indirect_thunk_rax" },
	{ 0x122c3a7e, "_printk" },
	{ 0xf0fdf6cb, "__stack_chk_fail" },
	{ 0x4e9ff7b0, "module_put" },
	{ 0xb3f985a8, "sg_alloc_table" },
	{ 0xacdf3914, "nvidia_p2p_put_pages_persistent" },
	{ 0x5b3f3e79, "nvidia_p2p_get_pages" },
	{ 0x5b8239ca, "__x86_return_thunk" },
	{ 0x642487ac, "nvidia_p2p_put_pages" },
	{ 0x2e8edfd2, "ib_register_peer_memory_client" },
	{ 0xbde5c050, "ib_unregister_peer_memory_client" },
	{ 0xfbe215e4, "sg_next" },
	{ 0xe671363, "nvidia_p2p_dma_map_pages" },
	{ 0x4454730e, "kmalloc_trace" },
	{ 0xdd3d0132, "nvidia_p2p_free_dma_mapping" },
	{ 0xf42ca687, "nvidia_p2p_free_page_table" },
	{ 0x3a0a8b87, "param_ops_int" },
	{ 0xb88db70c, "kmalloc_caches" },
	{ 0x970adefe, "nvidia_p2p_get_pages_persistent" },
	{ 0x2fa5cadd, "module_layout" },
};

MODULE_INFO(depends, "nvidia,ib_uverbs");


MODULE_INFO(srcversion, "B13C9DFD8CD4E8BE2B5D362");
