const std = @import("std");
const c = @cImport({
    @cDefine("GLFW_INCLUDE_NONE", {});
    @cInclude("vulkan/vulkan.h");
    @cInclude("GLFW/glfw3.h");
});

const Allocator = std.mem.Allocator;

const w = @import("glfw.zig");
const win = w.Window;

pub fn init(alloc: Allocator) !void {
    const window = try win.create(800, 600, "Vulkan");
    defer window.destroy();
    const extensions = w.getExtensions();

    // Instance --------------------------------------------------------------------
    const application_info: c.VkApplicationInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion = c.VK_API_VERSION_1_4,
        .engineVersion = c.VK_MAKE_VERSION(0, 0, 1),
        .pEngineName = "Block Engine",
        .applicationVersion = c.VK_MAKE_VERSION(0, 1, 0),
        .pApplicationName = "Vulkan App",
    };

    const instance_create_info: c.VkInstanceCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &application_info,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = null, // Validation layers can be set up later
        .enabledExtensionCount = @intCast(extensions.len),
        .ppEnabledExtensionNames = extensions.ptr,
    };

    var instance: c.VkInstance = undefined;
    _ = c.vkCreateInstance(&instance_create_info, null, &instance);
    defer c.vkDestroyInstance(instance, null);

    // Physical Device -----------------------------------------------------------
    var physical_device_count: u32 = undefined;
    _ = c.vkEnumeratePhysicalDevices(instance, &physical_device_count, null);
    std.debug.print("Physical Device Count :: {d}\n", .{physical_device_count});

    const physical_device_list = try alloc.alloc(c.VkPhysicalDevice, physical_device_count);
    defer alloc.free(physical_device_list);
    _ = c.vkEnumeratePhysicalDevices(instance, &physical_device_count, @ptrCast(physical_device_list));

    var j: u32 = 0;
    var found: bool = false;
    while (j < physical_device_count) {
        if (isSuitable(physical_device_list[j])) {
            //const physical_device: c.VkPhysicalDevice = physical_device_list[j];
            found = true;
            std.debug.print("Chosen ID :: {d} / {d}\n", .{ j + 1, physical_device_count });
        }
        j = j + 1;
    }

    if (!found) {
        return error.NO_VALID_GPU;
    }

    // Device ---------------------------------------------------------------------

}

// Helper Functions ---------------------------------------------------------------
fn isSuitable(device: c.VkPhysicalDevice) bool {
    var device_properties: c.VkPhysicalDeviceProperties = undefined;
    var device_features: c.VkPhysicalDeviceFeatures = undefined;
    _ = c.vkGetPhysicalDeviceProperties(device, &device_properties);
    _ = c.vkGetPhysicalDeviceFeatures(device, &device_features);

    var is_suitable: bool = undefined;
    if (device_properties.deviceType == c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU and device_features.geometryShader == 1) {
        is_suitable = true;
        std.debug.print("Chosen GPU :: {s}\n", .{device_properties.deviceName});
    } else {
        is_suitable = false;
    }

    return is_suitable;
}
