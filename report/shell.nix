{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  buildInputs = with pkgs; [
    typst
    tinymist
    typstyle

    texlab
  ];

  shellHook = '''';
}
