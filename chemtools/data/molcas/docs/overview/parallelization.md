<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/parallelization.html -->

[ ](<index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<index.html>)

1.1.4. Parallelization efforts for Molcas modules

[previous](<news.html> "1.1.3. New features and updates") | [next](<ack.html> "1.1.5. Acknowledgment") | [index](<genindex.html> "General Index")

# 1.1.4. Parallelization efforts for Molcas modules¶

Presented below is a table of modules in Molcas that _can_ benifit from parallel execution through distribution of work and/or resources. If a module is not listed in this table, and the module-specific documentation does not mention anything about parallelization, then you have to assume the module is not (efficiently) parallelized. This means that even though it will get executed in parallel, all processes will perform the same serial calculation! Be aware that for parallel modules with serial components, the use of the serial components (indirectly or through the use of a keyword) might adversely affect CPU and memory usage for large calculations. In that case, you might have to increase the runtime or memory, or avoid/use keywords that activate/deactivate the serial components.

Table 1.1.4.1 Modules in Molcas which benefit from parallel processing.¶ Module | Parallel speed-up expected for | Notable non-parallel parts  
---|---|---  
SEWARD |  conventional 2-el integrals Cholesky vectors |  1-el integrals Douglas–Kroll–Hess properties  
SCF |  orbital optimization |  properties  
RASSCF |  orbital optimization |  CI optimization properties  
MBPT2 |  |   
CASPT2 |  Cholesky vectors |  conventional 2-el integrals properties multi-state interaction  
ALASKA |  displacements (if using numerical gradients) |   
GEO |  displacements |   
  
### Table of Contents

  * [1\. Introduction](<intro.html>)
    * [1.1. Introduction to Molcas](<intro.html#introduction-to-molcas>)
      * [1.1.1. Molcas, Quantum Chemistry Software](<introduction.html>)
      * [1.1.2. The Molcas Manual](<aboutthismanual.html>)
      * [1.1.3. New features and updates](<news.html>)
      * 1.1.4. Parallelization efforts for Molcas modules
      * [1.1.5. Acknowledgment](<ack.html>)
      * [1.1.6. Citation for OpenMolcas and Molcas](<citation.html>)
      * [1.1.7. Web Site](<web.html>)
      * [1.1.8. Disclaimer](<disclaimer.html>)
  * [2\. Installation Guide](<installation.guide/ig.html>)
  * [3\. Short Guide to Molcas](<tutorials/tut.html>)
  * [4\. User’s Guide](<users.guide/ug.html>)
  * [5\. Advanced Examples and Annexes](<advanced.examples/ae.html>)



### Search

[previous](<news.html> "1.1.3. New features and updates") | [next](<ack.html> "1.1.5. Acknowledgment") | [index](<genindex.html> "General Index")

[Get PDF](<../Manual.pdf>) | [Show Source](<_sources/parallelization.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
