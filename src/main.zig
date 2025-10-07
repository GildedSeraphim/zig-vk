const std = @import("std");
const vk = @import("render/vulkan.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    _ = try vk.init(allocator);
}
