{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  buildInputs = with pkgs; [
    python312Packages.uv

    ruff
    ty
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.zlib}/lib:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"

    export UV_VENV_CLEAR=1

    uv venv
    source .venv/bin/activate
    uv pip compile pyproject.toml --extra dev -o requirements.lock
    uv pip sync requirements.lock
  '';
}
