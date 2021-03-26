# FastCorner with Halide
FAST corner detector with Halide.  

Not completed yet, but it can run and output same result as OpenCV. Time cost is the problem.

Because `select` in Halide evaluates both arguments, corner detection (less costly) before score calculation (very costly) seems meaningless. See [my question on stackoverflow](https://stackoverflow.com/questions/66792921/how-to-prevent-halide-select-from-evaluating-both-branches).

---

I have learned Halide and tried to utilize it in my CV projects for about 3 weeks. But I really can't find solution for some problems, the `if` statement (where I hope `select` can work) is just one of them. Moreover, it's really hard to find some useful documents or get some help.

Halide is an amazing project, and maybe someday in the future it will shine. But for now, I just give up.
