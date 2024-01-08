# Kaleidoscope
This is my work while following along with LLVM tutorials that can be found here: https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html. The tutorial covers three different three different sub-projects. I have included each in this repository under their own directory.
### toy
This project gives an example for how we can use our toy langauge to generate an object file that can be linked into a cpp file. By running `bash b`:
1. The clang compiler is used to transform kaleidoscope.cpp into an output kaleidoscope compiler
2. Then, we use the kaleidoscope compiler to convert the average.ks file into an object file output.o
3. Next, we use the clang compiler again to link output.o into a cpp file and generate an executable main
4. Lastly, we execute the main file and see successful use of our average function defined in average.ks
### toy_jit
This project leverages a jit compiler to create a command line / prompt interface for the toy language. By running `bash b`:
1. The clang compiler is used to transform kaleidoscope.cpp into a kaleidoscope executable
2. Running the kaliedoscope executable pulls us into the prompt interface to interact with the language
3. I have included a tests.txt file that you can copy and paste into the prompt to generate several different mandelbrot sets
4. crtl+d to exit the prompt interface
### toy_debug 
This project shows how llvm can also be used to generate debug information. By running `bash b`:
1. The clang compiler is used to transform kaleidoscope.cpp into a kaleidoscope debugger
2. Then, we pass the fib.ks file into the kaleidoscope debugger which emits thorough information regarding our .ks file
### toy_mlir
This project shows how llvm can be integrated with mlir based on the official mlir toy tutorial.
