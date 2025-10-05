const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{});
    const target = b.standardTargetOptions(.{});

    const glad_module = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true });
    glad_module.addIncludePath(b.path("ext/glad/"));
    const glad = b.addLibrary(.{ .name = "glad", .root_module = glad_module });

    glad_module.addCSourceFiles(.{ .files = &[_][]const u8{"ext/glad/glad.c"} });

    const exe = b.addExecutable(.{
        .name = "hello",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = b.graph.host,
            .optimize = optimize,
        }),
    });
    b.default_step.dependOn(&exe.step);

    // ### Libraries ###
    exe.linkSystemLibrary("vulkan");
    exe.linkSystemLibrary("dl");
    exe.linkSystemLibrary("pthread");
    exe.linkSystemLibrary("X11");
    exe.linkSystemLibrary("Xxf86vm");
    exe.linkSystemLibrary("Xrandr");
    exe.linkSystemLibrary("Xi");
    exe.linkSystemLibrary("glfw");
    exe.linkSystemLibrary("imgui");
    exe.linkSystemLibrary("cglm");
    exe.addIncludePath(b.path("ext/glad/"));
    exe.addIncludePath(b.path("ext/glad/glad/"));
    exe.linkLibrary(glad);
    // #################

    exe.linkLibCpp();
    b.installArtifact(exe);

    const run_exe = b.addRunArtifact(exe);
    const run_step = b.step("run", "Run the application");
    run_step.dependOn(&run_exe.step);
}
