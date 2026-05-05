{
  description = "penzai using uv2nix";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };

      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
      };

      pythonSets = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
            };
          };
          python = pkgs.python314;

          hacks = pkgs.callPackage pyproject-nix.build.hacks { };

          customOverlay = final: prev: {
            torch =
              (hacks.nixpkgsPrebuilt {
                from = pkgs.python314Packages.torchWithoutCuda;
                prev = prev.torch;
              }).overrideAttrs
                (old: {
                  passthru = (old.passthru or { }) // {
                    dependencies = lib.filterAttrs (
                      name: _: !(lib.hasPrefix "nvidia-" name || lib.hasPrefix "cuda-" name)
                    ) (old.passthru.dependencies or { });
                  };
                });
            torchvision =
              (hacks.nixpkgsPrebuilt {
                from = pkgs.python314Packages.torchvision;
                prev = prev.torchvision;
              }).overrideAttrs
                (old: {
                  passthru = (old.passthru or { }) // {
                    dependencies = lib.filterAttrs (
                      name: _: !(lib.hasPrefix "nvidia-" name || lib.hasPrefix "cuda-" name)
                    ) (old.passthru.dependencies or { });
                  };
                });
          };

        in
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
              overlay
              customOverlay
            ]
          )
      );

    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pythonSet = pythonSets.${system}.overrideScope editableOverlay;
          virtualenv = pythonSet.mkVirtualEnv "penzai-dev-env" workspace.deps.all;
        in
        {
          default = pkgs.mkShell {
            packages = [
              virtualenv
              pkgs.uv
              pkgs.nodejs_24
              pkgs.clinfo # GPU detection
              pkgs.opencl-headers
              pkgs.ocl-icd
              pkgs.intel-compute-runtime-legacy1
            ];
            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = pythonSet.python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
            };
            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT=$(git rev-parse --show-toplevel)
              alias npm='nix run .#npm --'
              alias npx='nix run .#npx --'
              alias jn='jupyter notebook'              
            '';
          };
        }
      );

      packages = forAllSystems (system: {
        default = pythonSets.${system}.mkVirtualEnv "penzai-env" workspace.deps.default;
      });
    };
}
