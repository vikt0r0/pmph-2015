CXX        = nvcc

SOURCES_CPP =Main.cu
HELPERS     =ScanHost.cu.h ScanKernels.cu.h MsspHost.cu.h MsspKernels.cu.h
EXECUTABLE  =testSolution


default: compile

.cu.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) -c $@ $<


compile: $(EXECUTABLE)
$(EXECUTABLE):
	$(CXX) -o $(EXECUTABLE) $(SOURCES_CPP)


run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
