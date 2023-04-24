# EVO projekt

Toto je readme k řešení projektu do předmětu EVO na téma kartézské genetické programování -- obrazové filtry.

Ve složce data se nachází výchozí obrázky, ze kterých jsem vycházel při experimentování.

Ve složce src se nachází tři Python skripty a Makefile, který nainstaluje s pomocí nástroje pip potřebné závislosti. Předpokládá se Python 3.7 (problematické u knihovny pro CGP při vyšších verzích z důvodů nekompatibility s novějšími s NumPy) -- buď je implicitně nainstalován, nebo také lze celkem snadno vytvořit virtuální prostředí. Pro spuštění projektu použijte main.py -h. Tento skript slouží ke spouštění hlavního algoritmu CGP. Dále lze použít skript plots.py, který v případě existence potřebných souborů vytvoří grafy pro závěrečnou obhajobu -- ty se objeví v hlavní složce.

Ve složce experimenty se nachází výsledky experimentů nutné pro vyplottování grafů ve skriptu plots.py. V případě zájmů o filtrované obrázky je buď možné spustit algoritmus, nebo napsat autorovi na mail xhudak03@vutbr.cz, který vám je zašle. (Oprava -- obrázky se vešly.) 

V této hlavní složce se pak nachází ještě dva soubory -- jeden pro pojednání v polovině semestru a jeden pro závěrečnou obhajobu.
