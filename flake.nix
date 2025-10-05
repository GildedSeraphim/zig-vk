{
  description = "Vulkan Flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    zig.url = "github:mitchellh/zig-overlay";
    zls.url = "github:zigtools/zls";
  };
  outputs =
    inputs@{
      self,
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        lib = nixpkgs.lib;
        zig = inputs.zig.packages.x86_64-linux.master;
        zls = inputs.zls.packages.x86_64-linux.default;
        pkgs = import nixpkgs {
          system = "${system}";
          config = {
            allowUnfree = true;
            nvidia.acceptLicense = true;
          };
        };
      in
      rec {
        devShells = {
          default = pkgs.mkShell rec {
            # REMOVE SHELLHOOK WHEN RUNNING FOR FIRST TIME

            #shellHook = ''
            #  unset NIX_CFLAGS_COMPILE
            # '';
            buildInputs = with pkgs; [
              ##################
              ### VULKAN SDK ###
              vulkan-headers
              vulkan-loader
              vulkan-validation-layers
              vulkan-tools
              vulkan-tools-lunarg
              vulkan-utility-libraries
              vulkan-extension-layer
              vulkan-volk
              vulkan-validation-layers
              spirv-headers
              spirv-tools
              spirv-cross
              mesa
              glslang
              ##################

              ####################
              ### Compat Tools ###
              xorg.libX11
              xorg.libXrandr
              xorg.libXcursor
              xorg.libXi
              xorg.libXxf86vm
              xorg.libXinerama
              wayland
              wayland-protocols
              kdePackages.qtwayland
              kdePackages.wayqt
              ####################

              #################
              ### Libraries ###
              imgui
              glfw
              glfw3
              glm
              cglm
              sdl3
              tinyobjloader
              vk-bootstrap
              vulkan-memory-allocator
              libGL
              libGLU
              #################

              #################
              ### Compilers ###
              shaderc
              gcc
              clang
              #################
            ];

            packages = with pkgs; [
              ### Langs ###
              zig
              zls
              #############

              #############
              ### Tools ###
              glfw
              glfw3
              renderdoc
              #############
            ];

            # LD_LIBRARY_PATH = "${lib.makeLibraryPath buildInputs}";
            #XDG_DATA_DIRS = builtins.getEnv "XDG_DATA_DIRS";
            #XDG_RUNTIME_DIR = "/run/user/1000";
            #STB_INCLUDE_PATH = "./headers/stb";
          };
        };
      }
    );
}
