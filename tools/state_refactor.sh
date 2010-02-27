#!/bin/bash
known_ca="\(all_label_counts\|confusion\|confusions\|emp_error\|errors\|history\|ndiscarded\|nfeatures\|null_errors\|predictions\|raw_predictions\|raw_values\|results\|selected_ids\|sensitivities\|sensitivity\|splits\|state[123]\|trained_confusion\|trained_confusions\|transerrors\|values\)"

sed -i \
 -e 's/\(\W\)State\.)/\1Stateful\./g' \
 -e 's/State\.__init/Stateful\.__init/g' \
 -e 's/State\.__str/Stateful\.__str/g' \
 -e 's/\.enableState/\.ca\.enable/g' \
 -e 's/\.enableStates/\.ca\.enable/g' \
 -e 's/\.enabledStates/\.ca\.enabled/g' \
 -e 's/\.disableState/\.ca\.disable/g' \
 -e 's/\.disableStates/\.ca\.disable/g' \
 -e 's/\.listStates/\.ca\.listing/g' \
 -e 's/\.hasState/\.ca\.isKnown/g' \
 -e 's/\.isStateEnabled/\.ca\.isEnabled/g' \
 -e 's/\._enableStatesTemporarily/\.ca\._enableTemporarily/g' \
 -e 's/\.isStateActive/\.ca\.isActive/g' \
 -e "s/\(\w\)[[]\([\"']\)$known_ca\2[]]/\1\.\3/g" \
 -e "s/self\._registerState(\([\"']\)$known_ca\1\,* */\2 = ConditionalAttribute(/g" \
 $@

exit 0

sed -i \
 -e 's/(State)/(Stateful)/g' \
 -e 's/import State/import ConditionalAttribute, Stateful/g' \
 -e "s/self\._registerState(\([\"']\)$known_ca\1\,* */\2 = ConditionalAttribute(/g" \
 $@

exit 0


obtained list of all conditional attributes ever known by
grep '_registerState(' *py `find ../mvpa -iname \*.py` 2>/dev/null| \
grep -v 'def _reg' |  sed -e "s/.*(\([\"']\)\([^ ,]*\)\1.*/\2/g"  | \
grep -v mvpa | sort | uniq | tr '\n' '\|'; echo

(State) Stateful

enableState  enable
enabledStates  enabled
listStates listing
ca items
_enableStatesTemporarily _enableTemporarily
_getRegisteredStates _getNames

hasState isKnown
isStateEnabled isEnabled
isStateActive isActive

listStates _getListing

GONE:
enableStates
disableStates
__enabledisableall
