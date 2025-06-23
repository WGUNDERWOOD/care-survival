# Heavily inspired by https://github.com/oliverevans96/maturin-nix-example
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane.url = "github:ipetkov/crane";
  };

  outputs = inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import inputs.nixpkgs {
        inherit system;
        overlays = [inputs.rust-overlay.overlays.default];
      };
      lib = pkgs.lib;

      customRustToolchain = pkgs.rust-bin.stable."1.86.0".default;
      craneLib =
        (inputs.crane.mkLib pkgs).overrideToolchain customRustToolchain;

      projectName =
        (craneLib.crateNameFromCargoToml {cargoToml = ./Cargo.toml;}).pname;
      projectVersion =
        (craneLib.crateNameFromCargoToml {
          cargoToml = ./Cargo.toml;
        })
        .version;

      pythonVersion = pkgs.python312;
      wheelTail = "cp312-cp312-linux_x86_64"; # change if pythonVersion changes
      wheelName = "${projectName}-${projectVersion}-${wheelTail}.whl";

      crateCfg = {
        src = craneLib.cleanCargoSource (craneLib.path ./.);
        nativeBuildInputs = [pythonVersion];
      };

      crateWheel =
        (craneLib.buildPackage (crateCfg
          // {
            pname = projectName;
            version = projectVersion;
          }))
        .overrideAttrs (old: {
          nativeBuildInputs = old.nativeBuildInputs ++ [pkgs.maturin];
          buildPhase =
            old.buildPhase
            + ''
              maturin build --offline --target-dir ./target
            '';
          installPhase =
            old.installPhase
            + ''
              cp target/wheels/${wheelName} $out/
            '';
        });
    in rec {
      packages = rec {
        default = crateWheel;
        pythonEnv =
          pythonVersion.withPackages
          (ps: [(lib.pythonPackage ps)] ++ (with ps; [ipython]));
      };

      lib = {
        pythonPackage = ps:
          ps.buildPythonPackage rec {
            pname = projectName;
            format = "wheel";
            version = projectVersion;
            src = "${crateWheel}/${wheelName}";
            doCheck = false;
            pythonImportsCheck = [projectName];
          };
      };

      devShells = rec {
        default = pkgs.mkShell {
          name = "env";
          src = ./.;
          nativeBuildInputs = [
            pkgs.alejandra
            pkgs.bacon
            pkgs.cargo
            pkgs.cargo-flamegraph
            pkgs.clippy
            pkgs.maturin
            pkgs.pkg-config
            pkgs.rust-analyzer
            pkgs.rustfmt
            packages.pythonEnv
          ];
        };
      };
    });
}
