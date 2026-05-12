orphan  

# Modern input reading

## What is presently there

Have a look in SUBROUTINE PAMINP this is the place where various input
reading routines are called and input sections are read.

This is also the place where the new input reading routine
read_menu_input() is called which reads the whole DIRAC.INP again.

The purpose of this documentation is to motivate the use of the new
read_menu_input() instead of the many input section reading routines and
to show how this can be done.

## Problems of the old input reading

- Similar and very lengthy code is repeated over and over again.
  Sometimes buggy code is copy-pasted over and over again.
- Keyword matching is realized using a table/goto scheme which is
  cumbersome and buggy to program and difficult to read and to extend
  with new keywords. It is difficult to see what code is activated by a
  specific keyword.
- No type checking is done: wrong inputs can result in wrong parameter
  initialization without segfault, infinite loop situations, or abort
  without any error message.
- Section reading routines are called twice, first time to initialize
  common block data (just in case they are not actually called, meaning
  in case the \*\*SECTION is not in the input) and possibly a second
  time to read input. The advantage of this is that options are
  initialized close to where they are read and set, but the price is a
  confusing top-level calling scheme.
- Case sensitive (caps lock).

## Features of the new input reading

- Like the old scheme: will segfault and print list of available
  keywords if unknown keyword found.
- Case insensitive.
- Some basic type checking: it will catch situations (segfault and point
  to the place) where user has given no or not enough arguments or
  arguments of unexpected type (integer instead of real).
- More compact code.
- No table/goto scheme, keywords are defined in one place only and added
  easily, without extending table parameters or other structures.
- It is easy to define aliases to sections for backwards compatibility
  (example: \*\*WAVE FUNCTION and \*\*METHOD can do the same thing
  without much code duplication).

## How does the new routine read the input

- The new input reading routine reads the whole input. The general
  structure (sectioning) is in one place. For sections and subsections
  individual routines are called.
- It ignores lines that start with "!" or "#".
- Lines that start with "." are matched against available keywords. How
  does it know which section it belongs to? The routine remembers the
  last line with a section label (last line that starts with "\*" or
  "\*\*").

This also means that this scheme favors:

    *SECTION
    .BLA

instead of the present:

    .SECTION
    *SECTION
    .BLA

However, the more compact scheme is not enforced.

## How to move a input section from old to new

- Have a look in the src/input/input_reader_sections.F90, add your
  section in subroutine read_input_sections and copy/paste/adapt one of
  the read_input_something subroutines.
- All keywords that belong to a section have to be between call
  reset_available_kw_list() and call check_whether_kw_found(word,
  kw_section), otherwise the user will not get the correct full list of
  available keywords.
- Inside if (kw_matches(word, '.FOOBAR')) then and end if you can do
  whatever you like but you can take advantage of type checking by using
  the kw_read subroutine interface. If the interface does not cover your
  problem, please extend the interface.
- The way the old scheme is set up you cannot simply remove a section
  and move it to the new one. You have to advance to the next section in
  the old routine, otherwise you get into an infinite loop. Search for
  move_to_next_star(). This is a workaround and as soon as everything is
  replaced, these calls can be removed (the whole old structure can be
  removed).
- If you use common blocks to save parameters then you have the problem
  of how to initialize them (in modules you can initialize them
  directly). I recommend to move your parameters to a module but if you
  insist you can use common blocks but then find a good way of
  initializing them. Please don't do it by calling section subroutines
  twice, this is confusing.
