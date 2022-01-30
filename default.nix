{ pkgs ? import <nixpkgs> {}
, stdenv ? pkgs.stdenv
}:
let
  local = rec {
    callPackage = pkgs.lib.callPackageWith collection;

    collection = rec {

      python = pkgs.python38Packages;
      mypython = pkgs.python38.withPackages (
        pp: let
          pyls = pp.python-language-server.override { providers=["pycodestyle"]; };
          pyls-mypy = pp.pyls-mypy.override { python-language-server=pyls; };

          altair-data-server = pp.buildPythonPackage rec {
            name = "altair-data-server";
            src = pp.fetchPypi {
              version = "0.4.1";
              pname = "altair_data_server";
              sha256 = "sha256:0azbkakgbjwxvkfsvdcw2vnpjn44ffwrqwsqzhh80rxjiaj0b4mk";
            };
            buildInputs = with pp; [ altair portpicker tornado jinja2 pytest ];
          };

          altair-viewer = pp.buildPythonPackage rec {
            name = "altair-viewer";
            src = pp.fetchPypi {
              version = "0.3.0";
              pname = "altair_viewer";
              sha256 = "sha256:0fa4ab233jbx11jfim35qys9yz769pdhkmfrvliyfnvwdggdnr19";
            };
            buildInputs = with pp; [ altair altair-data-server portpicker tornado
                                       jinja2 pytest ipython ];
          };

          altair-saver = pp.buildPythonPackage rec {
            name = "altair-saver";
            src = pp.fetchPypi {
              version = "0.5.0";
              pname = "altair_saver";
              sha256 = "sha256:15c7p23m8497jpvabg49bd858nsip31lv408n4fs2fwfhvvbr660";
            };
            # preConfigure = ''
            #   sed -i 's/"chrome"/"chromium"/g' ./altair_saver/savers/_selenium.py
            # '';
            propagatedBuildInputs = with pp; [ altair-viewer pkgs.nodejs altair-data-server
              altair portpicker tornado jinja2 pytest selenium
              pillow pypdf2];
          };

          bespon = pp.buildPythonPackage rec {
            pname = "bespon_py";
            version = "0.6.0";
            # propagatedBuildInputs = with mypython.pkgs ; [nr-types pyyaml];
            # doCheck = false; # local HTTP requests don't work
            src = pkgs.fetchFromGitHub {
              owner = "gpoore";
              repo = pname;
              rev = "183d0a49146025969266fc1b4157392d5ffda609";
              sha256 = "sha256:0x1ifklhh88fa6i693zgpb63646jxsyhj4j64lrvarckrb31wk23";
            };
          };

          codebraid = pp.buildPythonPackage rec {
            pname = "codebraid";
            version = "0.5.0";

            propagatedBuildInputs = [ bespon ];
            src = pkgs.fetchFromGitHub {
              owner = "gpoore";
              repo = pname;
              rev = "21ef9399918e750852e5aa70c443e44052250d78";
              sha256 = "sha256:05754y0rbj6qcm2r772r3asln8rp2n958mi04s29my18mrjqwdhn";
            };
          };

        in with pp; [
          ipython
          hypothesis
          pytest
          pytest-mypy
          pytest_xdist
          coverage
          codebraid
          pyls
          pyls-mypy
          pyyaml
          wheel
          multipledispatch
          graph-tool
          pygobject3
          ipdb
          scipy

          matplotlib
          pyqt5
          plotly
          altair
          altair-data-server
          altair-viewer
          altair-saver
          selenium

          # (projectq_ pp)
          pylatexenc
          ipywidgets

          pygraphviz
          pydot

          z3
          dataclasses-json
          sympy
          setuptools_scm
        ]);

      inherit (pkgs) cudatoolkit cudnn magma;

      shell = pkgs.mkShell {
        name = "shell";
        buildInputs = with pkgs; [
          mypython
          pandoc
          poppler
          poppler_utils
          cairo
          feh
          python.grip
          imagemagick
          gnome3.eog
          gobject-introspection
          gtk3
          gdb

          (let
             mytexlive = pkgs.texlive.override { python3=mypython; };
           in
             mytexlive.combine {
               scheme-medium = mytexlive.scheme-medium;
               inherit (mytexlive) fvextra upquote xstring pgfopts currfile
               collection-langcyrillic makecell ftnxtra minted catchfile framed
               pdflscape environ trimspaces mdframed zref needspace import
               beamerposter qcircuit xypic standalone preview amsmath thmtools
               tocloft tocbibind varwidth;
             }
          )
        ];
      shellHook = with pkgs; ''
        if test -f ./env.sh ; then
          . ./env.sh
          export QT_QPA_PLATFORM_PLUGIN_PATH=`echo ${pkgs.qt5.qtbase.bin}/lib/qt-*/plugins/platforms/`
        fi
      '';
      };

    };
  };
in
  local.collection



