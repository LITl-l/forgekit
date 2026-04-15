{
  description = "forgekit — research-incubator pipeline for local LLM development";

  inputs = {
    # Pinned to a specific rev so evaluation doesn't require hitting the
    # GitHub API on every `nix develop`. Bump by running `nix flake update`.
    nixpkgs.url = "github:NixOS/nixpkgs/13043924aaa7375ce482ebe2494338e058282925";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);
    in {
      devShells = forAllSystems (system:
        let
          isLinux = nixpkgs.lib.hasSuffix "linux" system;
          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = isLinux;
            };
          };
          # NOTE: We pin a CUDA 12.8 toolchain for host tooling (nvcc for custom
          # kernels, cudnn headers). The actual PyTorch / Triton wheels —
          # especially the sm_121 (GB10, Grace Blackwell) ones — must come from
          # `uv` / PyPI, not from nixpkgs. nixpkgs lags too far behind for
          # bleeding-edge arches.
          cudaPkgs = pkgs.cudaPackages_12_8 or pkgs.cudaPackages;
        in {
          default = pkgs.mkShell {
            name = "forgekit-dev";
            buildInputs = [
              pkgs.uv
              pkgs.python311
              pkgs.just
              # huggingface-cli is pulled via `uv sync` (huggingface_hub is a
              # transitive dep of the trl / transformers extras). Not installed
              # at the Nix layer because the nixpkgs build currently conflicts
              # with python3.11 (sphinx 9.1).
            ] ++ pkgs.lib.optionals isLinux [
              cudaPkgs.cudatoolkit
              cudaPkgs.cudnn
            ];
            shellHook = ''
              ${pkgs.lib.optionalString isLinux ''
                export CUDA_HOME=${cudaPkgs.cudatoolkit}
                export LD_LIBRARY_PATH=${cudaPkgs.cudatoolkit}/lib:${cudaPkgs.cudnn}/lib:$LD_LIBRARY_PATH
              ''}
              echo "forgekit dev shell — run: uv sync --extra dev"
            '';
          };
        });
    };
}
