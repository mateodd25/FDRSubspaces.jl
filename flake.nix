{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    nixpkgs-python.url = "github:cachix/nixpkgs-python";
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
  };

  nixConfig = {
    extra-trusted-public-keys =
      "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
  };

  outputs = { self, nixpkgs, devenv, systems, ... }@inputs:
    let forEachSystem = nixpkgs.lib.genAttrs (import systems);
    in {
      packages = forEachSystem (system: {
        devenv-up = self.devShells.${system}.default.config.procfileScript;
      });

      devShells = forEachSystem (system:
        let pkgs = nixpkgs.legacyPackages.${system};
        in {
          default = devenv.lib.mkShell {
            inherit inputs pkgs;
            modules = [{
              enterShell = ''
                echo "Entering FDRSubspaces.jl shell";
              '';
              packages = with pkgs; [
                stdenv.cc.cc.lib # required by jupyter
                gcc-unwrapped # fix: libstdc++.so.6: cannot open shared object file
                libz # fix: for numpy/pandas import
                gcc13 # newer gcc for CXXABI compatibility
                libgcc
              ];
              env.LD_LIBRARY_PATH =
                "${pkgs.gcc13.cc.lib}/lib64:${pkgs.gcc-unwrapped.lib}/lib64:${pkgs.libz}/lib";
              languages.julia.enable = true;
              # languages.julia.package = pkgs.julia_bin;
              languages.python = {
                enable = true;
                version = "3.12";
                venv = {
                  enable = true;

                  requirements = ./requirements.txt;
                };
              };
            }];
          };
        });
    };
}
