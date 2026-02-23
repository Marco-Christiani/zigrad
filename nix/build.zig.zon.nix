# generated for zigrad build.zig.zon dependencies

{
  linkFarm,
  fetchzip,
}:

linkFarm "zig-packages" [
  {
    name = "safetensors_zg-0.0.1-dRXUiGDtAAAS8GBHZT5IfFha7hG6KXCtVUuUxB-ob4Jc";
    path = fetchzip {
      url = "https://github.com/Marco-Christiani/safetensors-zg/archive/15787b35b541a4493630ec383750242eef422b64.tar.gz";
      hash = "sha256-AEf0Eero0NuUbH/R79ALXBtkbj8zWl859qV7RFNOQO0=";
    };
  }
  {
    name = "cova-0.10.1-_OE4R6vTBADvQugVWTn-HVjQmsMLh0yc2yp3g8oOOMGT";
    path = fetchzip {
      url = "https://github.com/Marco-Christiani/cova/archive/89b66faa69d5c8f2131245e6fd71dd1fa8d80351.tar.gz";
      hash = "sha256-aS6GchEReakH08IPB4YCdDVCMkfGd/XcfiJFWGJrlU8=";
    };
  }
]
