const std = @import("std");
const c = @cImport({
    @cDefine("GLFW_INCLUDE_NONE", {});
    @cInclude("vulkan/vulkan.h");
    @cInclude("GLFW/glfw3.h");
});

const Allocator = std.mem.Allocator;

const w = @import("glfw.zig");
const win = w.Window;

pub const BufferUsage = packed struct(u32) {
    transfer_src: bool = false,
    transfer_dst: bool = false,
    uniform_texel_buffer: bool = false,
    storage_texel_buffer: bool = false,
    uniform_buffer: bool = false,
    storage_buffer: bool = false,
    index_buffer: bool = false,
    vertex_buffer: bool = false,
    indirect_buffer: bool = false,
    _padding: enum(u23) { unset } = .unset,
};

pub const BufferFlags = packed struct(u32) {
    device_local: bool = false,
    host_visible: bool = false,
    host_coherent: bool = false,
    host_cached: bool = false,
    lazily_allocated: bool = false,
    _padding: enum(u27) { unset } = .unset,
};

pub fn init(alloc: Allocator) !void {
    const window = try win.create(800, 600, "Vulkan");
    defer window.destroy();
    const extensions = w.getExtensions();

    const vk_alloc_cb: ?*c.VkAllocationCallbacks = null;

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
    var surface_properties: c.VkSurfaceCapabilitiesKHR = undefined;
    _ = c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &surface_properties);

    var swapchain_size = c.VkExtent2D{};
    if (surface_properties.currentExtent.width == 0xFFFFFFFF) {
        swapchain_size.width = 800;
        swapchain_size.height = 600;
    } else {
        swapchain_size = surface_properties.currentExtent;
    }

    const swapchain_present_mode: c.VkPresentModeKHR = c.VK_PRESENT_MODE_FIFO_KHR;

    var desired_swapchain_images = surface_properties.minImageCount + 1;
    if ((surface_properties.maxImageCount > 0) and (desired_swapchain_images > surface_properties.maxImageCount)) {
        desired_swapchain_images = surface_properties.maxImageCount;
    }

    var pre_transform: c.VkSurfaceTransformFlagBitsKHR = undefined;
    if ((surface_properties.supportedTransforms & c.VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) != 0) {
        pre_transform = c.VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    } else {
        pre_transform = surface_properties.currentTransform;
    }

    var composite: c.VkCompositeAlphaFlagBitsKHR = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if ((surface_properties.supportedCompositeAlpha & c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) != 0) {
        composite = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    } else if ((surface_properties.supportedCompositeAlpha & c.VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) != 0) {
        composite = c.VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    } else if ((surface_properties.supportedCompositeAlpha & c.VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) != 0) {
        composite = c.VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
    } else if ((surface_properties.supportedCompositeAlpha & c.VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) != 0) {
        composite = c.VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;
    }

    const swapchain_create_info = c.VkSwapchainCreateInfoKHR{
        .sType = c.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = desired_swapchain_images,
        .imageFormat = c.VK_FORMAT_R8G8B8A8_SRGB,
        .imageColorSpace = c.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
        .imageExtent = swapchain_size,
        .imageArrayLayers = 1,
        .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = pre_transform,
        .compositeAlpha = composite,
        .presentMode = swapchain_present_mode,
        .clipped = c.VK_TRUE,
    };

    var swapchain: c.VkSwapchainKHR = undefined;
    _ = c.vkCreateSwapchainKHR(device, &swapchain_create_info, null, &swapchain);
    defer c.vkDestroySwapchainKHR(device, swapchain, null);

    // Images
    var swap_count: u32 = undefined;
    _ = c.vkGetSwapchainImagesKHR(device, swapchain, &swap_count, null);

    const images = try alloc.alloc(c.VkImage, swap_count);
    defer alloc.free(images);
    _ = c.vkGetSwapchainImagesKHR(device, swapchain, &swap_count, @ptrCast(images));

    const semaphore_create_info = c.VkSemaphoreCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    var semaphore: c.VkSemaphore = undefined;
    _ = c.vkCreateSemaphore(device, &semaphore_create_info, vk_alloc_cb, &semaphore);
    defer c.vkDestroySemaphore(device, semaphore, vk_alloc_cb);

    const fence_create_info = c.VkFenceCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = c.VK_FENCE_CREATE_SIGNALED_BIT,
    };
    var fence: c.VkFence = undefined;
    _ = c.vkCreateFence(device, &fence_create_info, vk_alloc_cb, &fence);
    defer c.vkDestroyFence(device, fence, vk_alloc_cb);

    var current_swap_image: u32 = undefined;
    _ = c.vkAcquireNextImageKHR(device, swapchain, 0xFFFFFFFF, semaphore, null, &current_swap_image); // need synchronization

    // Backbuffer -----------------------------------------------------------------
    const backbuffer_view_create_info = c.VkImageViewCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = images[0],
        .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
        .format = c.VK_FORMAT_R8G8B8A8_SRGB,
        .subresourceRange = .{
            .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    var backbuffer_view: c.VkImageView = undefined;
    _ = c.vkCreateImageView(device, &backbuffer_view_create_info, vk_alloc_cb, &backbuffer_view);
    defer c.vkDestroyImageView(device, backbuffer_view, vk_alloc_cb);

    // Queue ----------------------------------------------------------------------
    var queue: c.VkQueue = undefined;
    _ = c.vkGetDeviceQueue(device, 0, 0, &queue);
    // queues are destoyed with devices

    // Renderpass -----------------------------------------------------------------
    const attachment = c.VkAttachmentDescription{
        .format = c.VK_FORMAT_R8G8B8A8_SRGB,
        .samples = c.VK_SAMPLE_COUNT_1_BIT, // Not multisampled.
        .loadOp = c.VK_ATTACHMENT_LOAD_OP_CLEAR, // When starting the frame, we want tiles to be cleared.
        .storeOp = c.VK_ATTACHMENT_STORE_OP_STORE, // When ending the frame, we want tiles to be written out.
        .stencilLoadOp = c.VK_ATTACHMENT_LOAD_OP_DONT_CARE, // Don't care about stencil since we're not using it.
        .stencilStoreOp = c.VK_ATTACHMENT_STORE_OP_DONT_CARE, // Don't care about stencil since we're not using it.
        .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED, // The image layout will be undefined when the render pass begins.
        .finalLayout = c.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, // After the render pass is complete, we will transition to PRESENT_SRC_KHR layout.
    };

    const color_ref = c.VkAttachmentReference{ .attachment = 0, .layout = c.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

    const subpass = c.VkSubpassDescription{
        .pipelineBindPoint = c.VK_PIPELINE_BIND_POINT_GRAPHICS, // describes purpose of renderpass
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_ref,
    };

    const dependency = c.VkSubpassDependency{
        .srcSubpass = c.VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
    };

    const renderpass_create_info = c.VkRenderPassCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency,
    };

    var renderpass: c.VkRenderPass = undefined;
    _ = c.vkCreateRenderPass(device, &renderpass_create_info, vk_alloc_cb, &renderpass);
    defer c.vkDestroyRenderPass(device, renderpass, vk_alloc_cb);

    // Framebuffer ----------------------------------------------------------------
    const framebuffer_create_info = c.VkFramebufferCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass = renderpass,
        .attachmentCount = 1,
        .pAttachments = &backbuffer_view,
        .width = swapchain_size.width,
        .height = swapchain_size.height,
        .layers = 1,
    };
    var framebuffer: c.VkFramebuffer = undefined;
    _ = c.vkCreateFramebuffer(device, &framebuffer_create_info, vk_alloc_cb, &framebuffer);
    defer c.vkDestroyFramebuffer(device, framebuffer, vk_alloc_cb);

    // Descriptor Sets ------------------------------------------------------------
    const UBO_binding = c.VkDescriptorSetLayoutBinding{
        .binding = 0,
        .descriptorType = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = c.VK_SHADER_STAGE_ALL_GRAPHICS,
        .pImmutableSamplers = null,
    };
    const sampler_binding = c.VkDescriptorSetLayoutBinding{
        .binding = 1,
        .descriptorType = c.VK_DESCRIPTOR_TYPE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = c.VK_SHADER_STAGE_ALL_GRAPHICS,
        .pImmutableSamplers = null,
    };
    const image_binding = c.VkDescriptorSetLayoutBinding{
        .binding = 5,
        .descriptorType = c.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .descriptorCount = 1,
        .stageFlags = c.VK_SHADER_STAGE_FRAGMENT_BIT,
        .pImmutableSamplers = null,
    };

    const bindings = [_]c.VkDescriptorSetLayoutBinding{
        UBO_binding,
        sampler_binding,
        image_binding,
    };

    const descriptor_set_layout_create_info = c.VkDescriptorSetLayoutCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = @intCast(bindings.len),
        .pBindings = bindings[0..].ptr,
    };
    var descriptor_set_layout: c.VkDescriptorSetLayout = undefined;
    _ = c.vkCreateDescriptorSetLayout(device, &descriptor_set_layout_create_info, vk_alloc_cb, &descriptor_set_layout);
    defer c.vkDestroyDescriptorSetLayout(device, descriptor_set_layout, vk_alloc_cb);

    // Pipeline Layout ------------------------------------------------------------
    const pipeline_layout_create_info = c.VkPipelineLayoutCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = null,
    };
    var pipeline_layout: c.VkPipelineLayout = undefined;
    _ = c.vkCreatePipelineLayout(device, &pipeline_layout_create_info, vk_alloc_cb, &pipeline_layout);
    defer c.vkDestroyPipelineLayout(device, pipeline_layout, vk_alloc_cb);

    //Shaders ---------------------------------------------------------------------
    var frag_shader_module: c.VkShaderModule = undefined;
    var vert_shader_module: c.VkShaderModule = undefined;

    const vertex_shader_source =
        \\#version 330 core
        \\out vec3 FC;
        \\
        \\uniform vec2 r; // resolution
        \\
        \\void main() {
        \\    vec2 uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        \\    FC = vec3(uv, 0.0); // add .z for use in FC.rgb
        \\    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
        \\}
    ;
    const fragment_shader_source =
        \\#version 300 es
        \\precision highp float;
        \\
        \\uniform vec2 r;     
        \\uniform float t;    
        \\
        \\out vec4 o;        
        \\
        \\void main() {
        \\    vec3 FC = vec3(gl_FragCoord.xy, 0.5); 
        \\    vec3 rayDir = normalize(FC * 2.0 - r.xyy); 
        \\
        \\    vec4 colorAccum = vec4(0.0);
        \\    float z = 0.0;
        \\    float d = 0.0;
        \\
        \\    for (float i = 0.0; i < 80.0; i++) {
        \\        vec3 p = z * rayDir;
        \\        p.z += 6.0; // move camera back
        \\
        \\        vec3 a = normalize(cos(vec3(1.0, 2.0, 0.0) + t - d * 5.0));//decides turbulence
        \\
        \\        a = a * dot(a, p) - cross(a, p);
        \\
        \\        for (d = 1.0; d <= 2.0; d++) {
        \\            a += sin(a * d + t).yzx / d;
        \\        }
        \\
        \\        d = 0.05 * abs(length(p) - 3.0) + 0.04 * abs(a.y);
        \\        d = max(d, 1e-4); // safety against divide-by-zero
        \\        z += d;
        \\
        \\        vec4 col = (cos(d / 0.1 + vec4(1.0, z, z, 0.0)) + 1.0); //decides color
        \\        colorAccum += col / d * z;
        \\    }
        \\
        \\    o = vec4(tanh(colorAccum.rgb / 3e4), 1.0);
        \\}
    ;

    const frag_shader_module_create_info = c.VkShaderModuleCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = fragment_shader_source.len,
        .pCode = @ptrCast(@alignCast(fragment_shader_source)),
    };

    const vert_shader_module_create_info = c.VkShaderModuleCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vertex_shader_source.len,
        .pCode = @ptrCast(@alignCast(vertex_shader_source)),
    };

    _ = c.vkCreateShaderModule(device, &frag_shader_module_create_info, vk_alloc_cb, @ptrCast(&frag_shader_module));
    defer c.vkDestroyShaderModule(device, frag_shader_module, vk_alloc_cb);
    _ = c.vkCreateShaderModule(device, &vert_shader_module_create_info, vk_alloc_cb, @ptrCast(&vert_shader_module));
    defer c.vkDestroyShaderModule(device, vert_shader_module, vk_alloc_cb);

    // Graphics Pipeline ----------------------------------------------------------
    const fragment_shader_stage_info: c.VkPipelineShaderStageCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = c.VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = frag_shader_module,
        .pName = "main",
    };
    const vertex_shader_stage_info: c.VkPipelineShaderStageCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = c.VK_SHADER_STAGE_VERTEX_BIT,
        .module = vert_shader_module,
        .pName = "main",
    };

    const shader_stage_infos: []const c.VkPipelineShaderStageCreateInfo = &.{ vertex_shader_stage_info, fragment_shader_stage_info };

    const binding_description: c.VkVertexInputBindingDescription = .{
        .binding = 0,
        .stride = 0,
        .inputRate = c.VK_VERTEX_INPUT_RATE_VERTEX,
    };

    const attribute_description: c.VkVertexInputAttributeDescription = .{
        .location = 0,
        .binding = 0,
        .format = c.VK_FORMAT_R32G32B32_SFLOAT,
        .offset = 0,
    };

    const vertex_input_info = c.VkPipelineVertexInputStateCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexAttributeDescriptionCount = 1,
        .pVertexAttributeDescriptions = &attribute_description,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding_description,
    };

    const input_assembly_info: c.VkPipelineInputAssemblyStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = c.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = c.VK_FALSE,
    };

    const viewport: c.VkViewport = .{
        .x = 0.0,
        .y = 0.0,
        .width = @floatFromInt(swapchain_size.width),
        .height = @floatFromInt(swapchain_size.height),
        .minDepth = 0.0,
        .maxDepth = 1.0,
    };

    const scissor: c.VkRect2D = .{
        .offset = .{
            .x = 0.0,
            .y = 0.0,
        },
        .extent = swapchain_size,
    };

    const viewport_state_info: c.VkPipelineViewportStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    const rasterizer_info: c.VkPipelineRasterizationStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = c.VK_FALSE,
        .rasterizerDiscardEnable = c.VK_FALSE,
        .polygonMode = c.VK_POLYGON_MODE_FILL,
        .lineWidth = 1.0,
        .cullMode = c.VK_CULL_MODE_BACK_BIT,
        .frontFace = c.VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = c.VK_FALSE,
    };

    const multisampling_info: c.VkPipelineMultisampleStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .sampleShadingEnable = c.VK_FALSE,
        .rasterizationSamples = c.VK_SAMPLE_COUNT_1_BIT,
    };

    const color_blend_attachment: c.VkPipelineColorBlendAttachmentState = .{
        .colorWriteMask = c.VK_COLOR_COMPONENT_R_BIT | c.VK_COLOR_COMPONENT_G_BIT | c.VK_COLOR_COMPONENT_B_BIT | c.VK_COLOR_COMPONENT_A_BIT,
        .blendEnable = c.VK_TRUE,
        .srcColorBlendFactor = c.VK_BLEND_FACTOR_SRC_ALPHA,
        .dstColorBlendFactor = c.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .colorBlendOp = c.VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = c.VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = c.VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = c.VK_BLEND_OP_ADD,
    };

    const color_blend_info: c.VkPipelineColorBlendStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = c.VK_FALSE,
        .logicOp = c.VK_LOGIC_OP_COPY,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment,
        .blendConstants = .{ 0.0, 0.0, 0.0, 0.0 },
    };

    const pipeline_info: c.VkGraphicsPipelineCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = @intCast(shader_stage_infos.len),
        .pStages = shader_stage_infos.ptr,
        .pVertexInputState = &vertex_input_info,
        .pInputAssemblyState = &input_assembly_info,
        .pViewportState = &viewport_state_info,
        .pRasterizationState = &rasterizer_info,
        .pMultisampleState = &multisampling_info,
        .pDepthStencilState = null,
        .pColorBlendState = &color_blend_info,
        .pDynamicState = null,
        .layout = pipeline_layout,
        .renderPass = renderpass,
        .subpass = 0,
        .basePipelineHandle = null,
        .basePipelineIndex = -1,
    };

    var pipeline: c.VkPipeline = undefined;
    _ = c.vkCreateGraphicsPipelines(device, null, 1, &pipeline_info, vk_alloc_cb, &pipeline);
    defer c.vkDestroyPipeline(device, pipeline, vk_alloc_cb);

    //Descriptor Pool -------------------------------------------------------------
    const descriptor_pool_size = c.VkDescriptorPoolSize{
        .type = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
    };

    const descriptor_pool_create_info = c.VkDescriptorPoolCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &descriptor_pool_size,
    };

    var descriptor_pool: c.VkDescriptorPool = undefined;
    _ = c.vkCreateDescriptorPool(device, &descriptor_pool_create_info, vk_alloc_cb, &descriptor_pool);
    defer c.vkDestroyDescriptorPool(device, descriptor_pool, vk_alloc_cb);

    // Descriptor Set -------------------------------------------------------------
    const descriptor_set_alloc_info = c.VkDescriptorSetAllocateInfo{
        .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorSetCount = 1,
        .descriptorPool = descriptor_pool,
        .pSetLayouts = &descriptor_set_layout,
    };

    var descriptor_set: c.VkDescriptorSet = undefined;
    _ = c.vkAllocateDescriptorSets(device, &descriptor_set_alloc_info, &descriptor_set);

    // Memory ---------------------------------------------------------------------
    const buffer_usage = BufferUsage{
        .uniform_buffer = true,
        .transfer_dst = true,
    };

    const buffer_flags = BufferFlags{
        .device_local = true,
    };

    const queue_family: u32 = 0;

    const buffer_create_info = c.VkBufferCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = 64,
        .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
        .usage = @bitCast(buffer_usage),
        .queueFamilyIndexCount = 1,
        .pQueueFamilyIndices = &queue_family,
    };

    var buffer: c.VkBuffer = undefined;
    _ = c.vkCreateBuffer(device, &buffer_create_info, vk_alloc_cb, &buffer);
    defer c.vkDestroyBuffer(device, buffer, vk_alloc_cb);

    var memory_req: c.VkMemoryRequirements = undefined;
    _ = c.vkGetBufferMemoryRequirements(device, buffer, &memory_req);

    var memory_properties: c.VkPhysicalDeviceMemoryProperties = undefined;
    _ = c.vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    const alloc_info = c.VkMemoryAllocateInfo{
        .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memory_req.size,
        .memoryTypeIndex = pick_memory_type(memory_properties, memory_req.memoryTypeBits, @bitCast(buffer_flags)),
    };

    var device_memory: c.VkDeviceMemory = undefined;
    _ = c.vkAllocateMemory(device, &alloc_info, vk_alloc_cb, &device_memory);
    _ = c.vkBindBufferMemory(device, buffer, device_memory, 0);
    defer c.vkFreeMemory(device, device_memory, null);

    var data: [*c]u8 = undefined;
    _ = c.vkMapMemory(device, device_memory, 0, c.VK_WHOLE_SIZE, 0, @ptrCast(&data));
    defer c.vkUnmapMemory(device, device_memory);

    const descriptor_buffer_info = c.VkDescriptorBufferInfo{
        .buffer = buffer,
        .offset = 0,
        .range = 64,
    };

    const write_descriptor_set = c.VkWriteDescriptorSet{
        .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptor_set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &descriptor_buffer_info,
    };

    _ = c.vkUpdateDescriptorSets(device, 1, &write_descriptor_set, 0, null);

    // Command Pools --------------------------------------------------------------
    const command_pool_create_info = c.VkCommandPoolCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = 0,
    };

    var command_pool: c.VkCommandPool = undefined;
    _ = c.vkCreateCommandPool(device, &command_pool_create_info, vk_alloc_cb, &command_pool);
    defer c.vkDestroyCommandPool(device, command_pool, vk_alloc_cb);

    // Command Buffer -------------------------------------------------------------
    const command_buffer_alloc_info = c.VkCommandBufferAllocateInfo{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandBufferCount = 1,
        .commandPool = command_pool,
        .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    };

    var command_buffer: c.VkCommandBuffer = undefined;
    _ = c.vkAllocateCommandBuffers(device, &command_buffer_alloc_info, &command_buffer);
    defer c.vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);

    // Now we can render!
    const begin_info: c.VkCommandBufferBeginInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    _ = c.vkBeginCommandBuffer(command_buffer, &begin_info);

    const clear_color: c.VkClearValue = .{ .color = .{ .float32 = .{ 0.0, 0.0, 0.0, 1.0 } } };
    const renderpass_begin_info = c.VkRenderPassBeginInfo{
        .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = renderpass,
        .framebuffer = framebuffer,
        .renderArea = .{
            .extent = swapchain_size,
            .offset = .{ .x = 0, .y = 0 },
        },
        .clearValueCount = 1,
        .pClearValues = &clear_color,
    };

    _ = c.vkCmdBeginRenderPass(command_buffer, &renderpass_begin_info, c.VK_SUBPASS_CONTENTS_INLINE);
    _ = c.vkCmdBindPipeline(command_buffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    _ = c.vkCmdBindDescriptorSets(command_buffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 0, null);
    _ = c.vkCmdSetViewport(command_buffer, 0, 1, &viewport);
    _ = c.vkCmdDraw(command_buffer, 3, 1, 0, 0);
    _ = c.vkCmdEndRenderPass(command_buffer);
    _ = c.vkEndCommandBuffer(command_buffer);

    // Submit ---------------------------------------------------------------------
    const submit_info = c.VkSubmitInfo{
        .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
    };
    _ = c.vkQueueSubmit(queue, 1, &submit_info, null);

    // Present here
    const present_info = c.VkPresentInfoKHR{
        .sType = c.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &semaphore,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = @ptrCast(&images),
        .pResults = null,
    };

    _ = c.vkQueuePresentKHR(queue, &present_info);
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

pub fn pick_memory_type(memory_properties: c.VkPhysicalDeviceMemoryProperties, type_bits: u32, flags: u32) u32 {
    var memory_type_index: u32 = 0;
    for (0..memory_properties.memoryTypeCount) |index| {
        const memory_type = memory_properties.memoryTypes[index];

        if (((type_bits & (@as(u64, 1) << @intCast(index))) != 0) and (memory_type.propertyFlags & flags) != 0 and (memory_type.propertyFlags & c.VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD) == 0) {
            memory_type_index = @intCast(index);
        }
    }

    return memory_type_index;
}
