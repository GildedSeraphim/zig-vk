const std = @import("std");
const c = @import("../clibs.zig");

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
        .pNext = null,
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
    features: c.VkPhysicalDeviceFeatures,

    pub fn pickPhysicalDevice(a: Allocator, i: Instance) !PhysicalDevice {
        var device_count: u32 = 0;
        mapError(try c.vkEnumeratePhysicalDevices(i.handle, &device_count, null));
        const devices_list = try a.alloc(c.VkPhysicalDevice, device_count);
        defer a.free(devices_list);
        mapError(try c.vkEnumeratePhysicalDevices(i.handle, &device_count, @ptrCast(devices_list)));

        return PhysicalDevice{ .handle = devices_list[0] };
    }

    pub fn getDeviceFeatures(physical_device: PhysicalDevice) !PhysicalDevice {
        const physical_device_features: c.VkPhysicalDeviceFeatures2 = .{
            .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        };
        mapError(try c.vkGetPhysicalDeviceFeatures(physical_device.handle, &physical_device_features));

        return PhysicalDevice{
            .features = physical_device_features,
        };
    }

    pub fn createDevice(physical_device: PhysicalDevice) !c.VkDevice {
        const device_info: c.VkDeviceCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = &physical_device.features,
            .queueCreateInfoCount = 0,
            .pQueueCreateInfos = null, //Setup queues
            .enabledExtensionCount = @intCast(device_extensions.len),
            .ppEnabledExtensionNames = device_extensions.ptr, //Add extensions
            .pEnabledFeatures = null, //get physical device features
        };

        var device: c.VkDevice = undefined;

        mapError(try c.vkCreateDevice(
            physical_device.handle,
            &device_info,
            null,
            &device,
        ));
    }

    pub fn destroy(self: PhysicalDevice) !void {}
};

pub const Device = struct {
    handle: c.VkDevice,
};
