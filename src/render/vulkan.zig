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

    const validation_layers: []const [*c]const u8 = &[_][*c]const u8{
        "VK_LAYER_KHRONOS_validation",
    };

    const device_extensions: []const [*c]const u8 = &.{
        "VK_KHR_swapchain",
    };

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
        .enabledLayerCount = @intCast(validation_layers.len),
        .ppEnabledLayerNames = validation_layers.ptr, // Validation layers can be set up later
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
    var physical_device: c.VkPhysicalDevice = undefined;
    while (j < physical_device_count) {
        if (isSuitable(physical_device_list[j])) {
            physical_device = physical_device_list[j];
            found = true;
            std.debug.print("Chosen ID :: {d} / {d}\n", .{ j + 1, physical_device_count });
        }
        j = j + 1;
    }

    if (!found) {
        return error.NO_VALID_GPU;
    }

    std.debug.print("{any}\n", .{@as([*]u64, @ptrCast(@alignCast(physical_device)))[0..4]}); // Luccie madness, code used by the deranged.

    // Surface ---------------------------------------------------------------------
    var surface: c.VkSurfaceKHR = undefined;
    _ = c.glfwCreateWindowSurface(instance, @ptrCast(window.raw), null, &surface);
    defer c.vkDestroySurfaceKHR(instance, surface, null);

    // Device ----------------------------------------------------------------------
    const priorities = [_]f32{1.0};

    const queue_create_info = c.VkDeviceQueueCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .queueFamilyIndex = 0,
        .queueCount = 1,
        .pQueuePriorities = &priorities[0],
    };

    const physical_device_features = c.VkPhysicalDeviceFeatures{
        .geometryShader = c.VK_TRUE,
        .tessellationShader = c.VK_TRUE,
    };

    const dynamic_rendering_features = c.VkPhysicalDeviceDynamicRenderingFeatures{
        .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
        .pNext = null,
        .dynamicRendering = c.VK_TRUE,
    };

    const device_create_info = c.VkDeviceCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &dynamic_rendering_features,
        .flags = 0,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_create_info,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = null,
        .enabledExtensionCount = @as(u32, @intCast(device_extensions.len)),
        .ppEnabledExtensionNames = device_extensions.ptr,
        .pEnabledFeatures = &physical_device_features,
    };

    var device: c.VkDevice = undefined;
    const result = c.vkCreateDevice(physical_device, &device_create_info, null, &device);
    if (result != c.VK_SUCCESS) {
        std.debug.print("vkCreateDevice failed: {d}\n", .{result});
        return error.DeviceCreationFailed;
    }
    defer c.vkDestroyDevice(device, null);

    // Swapchain ------------------------------------------------------------------
    const extent = c.VkExtent2D{ .width = 800, .height = 600 };
    const swapchain_create_info = c.VkSwapchainCreateInfoKHR{
        .sType = c.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = 4,
        .imageFormat = c.VK_FORMAT_R8G8B8A8_SRGB,
        .imageColorSpace = c.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .presentMode = c.VK_PRESENT_MODE_IMMEDIATE_KHR,
        .imageSharingMode = c.VK_SHARING_MODE_CONCURRENT,
        .queueFamilyIndexCount = 1,
    };

    var swapchain: c.VkSwapchainKHR = undefined;
    _ = c.vkCreateSwapchainKHR(device, &swapchain_create_info, null, &swapchain);
    defer c.vkDestroySwapchainKHR(device, swapchain, null);

    const images = try alloc.alloc(c.VkImage, 4);
    var swap_count: u32 = undefined;
    defer alloc.free(images);
    _ = c.vkGetSwapchainImagesKHR(device, swapchain, &swap_count, @ptrCast(images));

    std.debug.print("Swap Count :: {d}\n", .{swap_count});
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

fn getQueueFamilyProperties(phys_dev: c.VkPhysicalDevice, alloc: Allocator) ![]const c.VkQueueFamilyProperties {
    var count: u32 = undefined;
    _ = c.vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &count, null);

    const queue_list = try alloc.alloc(c.VkQueueFamilyProperties, count);

    _ = c.vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &count, @ptrCast(queue_list));

    std.debug.print("Queue count:: {d}\n", .{count});
    return queue_list;
}

fn graphicsQueue(phys_dev: c.VkPhysicalDevice, alloc: Allocator) !u32 {
    const queue_families = try getQueueFamilyProperties(phys_dev, alloc);
    defer alloc.free(queue_families);

    var graphics_queue: ?u32 = null;

    for (queue_families, 1..) |family, index| {
        if (graphics_queue) |_| {
            break;
        }

        if ((family.queueFlags & c.VK_QUEUE_GRAPHICS_BIT) != 0x0) {
            graphics_queue = @intCast(index);
        }
    }

    std.debug.print("Graphics Queue Index:: {d}\n", .{graphics_queue.?});

    return graphics_queue.?;
}

fn presentQueue(phys_dev: c.VkPhysicalDevice, alloc: Allocator, surface: c.VkSurfaceKHR) !u32 {
    const queue_families = try getQueueFamilyProperties(phys_dev, alloc);
    defer alloc.free(queue_families);

    var present_queue: ?u32 = null;

    for (queue_families, 0..) |_, index| {
        if (present_queue) |_| {
            break;
        }

        var support: u32 = undefined;
        _ = c.vkGetPhysicalDeviceSurfaceSupportKHR(phys_dev, @intCast(index), surface, &support);

        if (support == c.VK_TRUE) {
            present_queue = @intCast(index);
        }
    }

    std.debug.print("Present Queue Index:: {d}\n", .{present_queue.?});

    return present_queue.?;
}
