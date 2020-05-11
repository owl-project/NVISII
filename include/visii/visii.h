/**
   A contrived example of ignoring too many commands in one comment.

   @forcpponly
   This is C++-specific.
   @endforcpponly

   @beginPythonOnly
   This is specific to @b Python.
   @endPythonOnly

   @transferfull Command ignored, but anything here is still included.

   @compileroptions This function must be compiled with /EHa when using MSVC.
*/
int Contrived() {return 1;}