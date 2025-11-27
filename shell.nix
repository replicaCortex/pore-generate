{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  buildInputs = with pkgs; [
    python312Packages.uv

    ruff
    ty

    zlib
    gcc
  ];

  LD_LIBRARY_PATH = "${pkgs.zlib}/lib:$LD_LIBRARY_PATH:${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH";
  UV_VENV_CLEAR = 1;

  shellHook = ''
    uv venv
    source .venv/bin/activate
    uv pip compile pyproject.toml --extra dev -o requirements.lock
    uv pip sync requirements.lock
  '';
}
