@ECHO OFF
SET python=%1
SET pypi_index=%2
IF     %python%""     == "" SET python=python
IF     %pypi_index%"" == "" SET pypi_index=https://pypi.vnpy.com
IF NOT %pypi_index%"" == "" SET pypi_index=--index-url %pypi_index%
@ECHO ON

%python% -m pip install --extra-index-url https://pypi.vnpy.com vnpy_ctastrategy
%python% -m pip install --extra-index-url https://pypi.vnpy.com vnpy_ctabacktester
%python% -m pip install --extra-index-url https://pypi.vnpy.com vnpy_datamanager
%python% -m pip install --extra-index-url https://pypi.vnpy.com vnpy_sqlite
%python% -m pip install --extra-index-url https://pypi.vnpy.com vnpy_ctp
%python% -m pip install --extra-index-url https://pypi.vnpy.com vnpy_datarecorder
%python% -m pip install --extra-index-url https://pypi.vnpy.com vnpy_spreadtrading