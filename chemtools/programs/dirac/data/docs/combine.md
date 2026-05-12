orphan  

# Combining other methods with ECP

In DIRAC program, ECP can be incorporated with several ground and
excited state calculation methods. The calculation methods can be set in
input file. Below is the few examples of hydrogen iodide calculations.
(See `ecp_input` for the molecular input)

## DFT calculation

See the quick Bi2 molecule test (`DFT.inp <../../../test/ecp/DFT.inp>`
and `Bi2.xyz <../../../test/ecp/Bi2.xyz>`)

<div class="literalinclude">

../../../test/ecp/DFT.inp

</div>

## COSCI calculation

<div class="literalinclude">

COSCI.inp

</div>

## MP2 and CC calculation

<div class="literalinclude">

CC.inp

</div>
