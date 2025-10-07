const std = @import("std");
const vk = @import("render/vulkan.zig");

pub fn main() !void {
    std.debug.print("Hello World!", .{});
    _ = try vk.init();
}
