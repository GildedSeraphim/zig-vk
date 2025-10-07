const std = @import("std");
const c = @cImport({
    @cDefine("GLFW_INCLUDE_NONE", {});
    @cInclude("vulkan/vulkan.h");
    @cInclude("GLFW/glfw3.h");
});

pub fn init() !void {
    _ = c.glfwInit();
}
