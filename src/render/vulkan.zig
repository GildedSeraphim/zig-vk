const std = @import("std");
const c = @import("../c.zig");
const window = @import("./window.zig");

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
    device_lost,
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
        c.VK_ERROR_DEVICE_LOST => Error.device_lost,
        else => Error.unknown_error,
    };
}

pub const Instance = struct {
    handle: c.VkInstance,

    pub fn create() !Instance {
        const app_info: c.VkApplicationInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = null,
            .pApplicationName = "Vulkan Renderer",
            .applicationVersion = c.VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "vk",
            .engineVersion = c.VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = c.VK_MAKE_VERSION(1, 3, 0),
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
        mapError(try c.vkCreateInstance(&create_info, null, &instance));

        return Instance{
            .handle = instance,
        };
    }

    pub fn destroy(self: Instance) !void {
        mapError(try c.vkDestroyInstance(self.handle, null));
    }
};

pub const Device = struct {
    handle: c.VkDevice,

    pub fn destroyDevice(self: Device) !void {
        mapError(try c.vkDestroyDevice(self.handle, null));
    }
};

pub const PhysicalDevice = struct {
    handle: c.VkPhysicalDevice,

    pub fn pickDevice(allocator: Allocator, instance: Instance) !PhysicalDevice {
        // Changes physical_device_count to the number of physical devices available
        var physical_device_count: u32 = undefined;
        mapError(try c.vkEnumeratePhysicalDevices(instance.handle, &physical_device_count, null));
        //

        // Fills physical_devices with the c.VkPhysicalDevice available
        const physical_devices = try allocator.alloc(c.VkPhysicalDevice, physical_device_count);
        mapError(try c.vkEnumeratePhysicalDevices(instance.handle, &physical_device_count, &physical_devices));
        //

        return PhysicalDevice{
            //simply picks the first device for now we can
            //add parameters later by analyzing properties
            .handle = physical_devices[0],
        };
    }

    pub fn queueFamilyProperties(self: PhysicalDevice, allocator: Allocator) ![]const c.VkQueueFamilyProperties {
        var q_fam_count: u32 = undefined;
        mapError(try c.vkGetPhysicalDeviceQueueFamilyProperties(self.handle, &q_fam_count, null));

        const q_properties = allocator.alloc(c.VkQueueFamilyProperties, q_fam_count);
        mapError(try c.vkGetPhysicalDeviceQueueFamilyProperties(self.handle, &q_fam_count, &q_properties));

        return q_properties;
    }

    pub fn createDevice(self: PhysicalDevice) !Device {
        const device_queue_create_info: c.VkDeviceQueueCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = null,
            .queueFamilyIndex = 0,
            .queueCount = 1,
            .pQueuePriorities = 1.0,
        };

        const create_info: c.VkDeviceCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = null,
            .queueCreateInfoCount = 0,
            .pQueueCreateInfos = &device_queue_create_info,
            .enabledExtensionCount = @intCast(device_extensions.len),
            .ppEnabledExtensionNames = device_extensions.ptr,
            .pEnabledFeatures = null,
        };

        const device: c.VkDevice = undefined;
        mapError(try c.vkCreateDevice(self.handle, &create_info, null, &device));

        // We have now created our device
        // It can be used in the following ways ::
        // - The creation of queues (which we will be doing here)
        // - Creation and tracking of synchronization constructs
        // - Allocating, freeing, and managing memory
        // - Creation and destruction of command buffers / buffer pools
        // - Creation, destruction, and management of the graphics state (Pipelines, resource descriptors)

        return Device{
            .handle = device,
        };
    }
};

pub const Buffer = struct {
    handle: c.VkCommandBuffer,
};
