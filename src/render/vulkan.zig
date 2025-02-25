const std = @import("std");
const c = @import("../clibs.zig");
const expect = std.testing.expect;

const Allocator = std.mem.Allocator;

const builtin = @import("builtin");
const debug = (builtin.mode == .Debug);

const validation_layers: []const [*c]const u8 = if (!debug) &[0][*c]const u8{} else &[_][*c]const u8{
    "VK_LAYER_KHRONOS_validation",
};

const device_extensions: []const [*c]const u8 = &[_][*c]const u8{
    c.VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

pub const Error = error{
    out_of_host_memory,
    out_of_device_memory,
    initialization_failed,
    layer_not_present,
    extension_not_present,
    feature_not_present,
    too_many_objects,
    device_lost,
    incompatible_driver,
    unknown_error,
};

pub fn mapError(result: c_int) !void {
    return switch (result) {
        c.VK_SUCCESS => {},
        c.VK_ERROR_OUT_OF_HOST_MEMORY => Error.out_of_host_memory,
        c.VK_ERROR_OUT_OF_DEVICE_MEMORY => Error.out_of_device_memory,
        c.VK_ERROR_INITIALIZATION_FAILED => Error.initialization_failed,
        c.VK_ERROR_LAYER_NOT_PRESENT => Error.layer_not_present,
        c.VK_ERROR_EXTENSION_NOT_PRESENT => Error.extension_not_present,
        c.VK_ERROR_FEATURE_NOT_PRESENT => Error.feature_not_present,
        c.VK_ERROR_TOO_MANY_OBJECTS => Error.too_many_objects,
        c.VK_ERROR_DEVICE_LOST => Error.device_lost,
        c.VK_ERROR_INCOMPATIBLE_DRIVER => Error.incompatible_driver,
        else => Error.unknown_error,
    };
}

pub const Instance = struct {
    handle: c.VkInstance,

    const app_info: c.VkApplicationInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = null,
        .pApplicationName = "Voxel engine",
        .applicationVersion = c.VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "sabr engine",
        .engineVersion = c.VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = c.VK_API_VERSION_1_3,
    };

    const create_info: c.VkInstanceCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = @intCast(validation_layers.len),
        .ppEnabledLayerNames = validation_layers.ptr,
        .enabledExtensionCount = @intCast(device_extensions.len),
        .ppEnabledExtensionNames = device_extensions.ptr,
    };

    var instance: c.VkInstance = undefined;

    pub fn create() !Instance {
        mapError(try c.vkCreateInstance(&create_info, null, &instance));
        return Instance{
            .handle = instance,
        };
    }

    pub fn destroy(i: Instance) void {
        mapError(try c.vkDestroyInstance(i.handle, null));
    }
};

pub const PhysicalDevice = struct {
    handle: c.VkPhysicalDevice,
    // We do not create physical devices, we enumerate them
    // This gives us a list of devices and we can then filter them
    // for various properties. For now we will just pick the first
    // device
    pub fn pickPhysDevices(allocator: Allocator, instance: Instance) !PhysicalDevice {
        var count: u32 = undefined;
        mapError(try c.vkEnumeratePhysicalDevices(instance.handle, &count, null));

        const physical_devices = try allocator.alloc(c.VkPhysicalDevice, count);
        defer allocator.free(physical_devices);

        mapError(try c.vkEnumeratePhysicalDevices(instance.handle, &count, physical_devices));
        const physical_device = physical_devices[0];

        // In the future we may want to use parameters to
        // select the physical device. This can be done by
        // vkEnumerateDeviceExtensionProperties and with
        // vkGetPhysicalDeviceProperties.

        return PhysicalDevice{
            .handle = physical_device,
        };
    }
};

pub const Device = struct {
    handle: c.VkDevice,

    pub fn createDevice(physical_device: PhysicalDevice) !Device {
        const priority: f32 = 1.0;
        const queue_create_info: c.VkDeviceQueueCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = 0,
            .queueCount = 1,
            .pQueuePriorities = &priority,
        };

        const create_info: c.VkDeviceCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue_create_info,
            .enabledExtensionCount = @intCast(device_extensions.len),
            .ppEnabledExtensionNames = device_extensions.ptr,
        };

        var device: c.VkDevice = undefined;
        mapError(try c.vkCreateDevice(physical_device.handle, &create_info, null, &device));

        return Device{
            .handle = device,
        };
    }
};
