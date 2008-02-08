#!/bin/bash
known_states="\(all_label_counts\|confusion\|confusions\|emp_error\|errors\|history\|ndiscarded\|nfeatures\|null_errors\|predictions\|raw_predictions\|raw_values\|results\|selected_ids\|sensitivities\|sensitivity\|splits\|state[123]\|trained_confusion\|trained_confusions\|transerrors\|values\)"

sed -i \
 -e 's/\(\W\)State\.)/\1Stateful\./g' \
 -e 's/State\.__init/Stateful\.__init/g' \
 -e 's/State\.__str/Stateful\.__str/g' \
 -e 's/\.enableState/\.states\.enable/g' \
 -e 's/\.enableStates/\.states\.enable/g' \
 -e 's/\.enabledStates/\.states\.enabled/g' \
 -e 's/\.disableState/\.states\.disable/g' \
 -e 's/\.disableStates/\.states\.disable/g' \
 -e 's/\.listStates/\.states\.listing/g' \
 -e 's/\.hasState/\.states\.isKnown/g' \
 -e 's/\.isStateEnabled/\.states\.isEnabled/g' \
 -e 's/\._enableStatesTemporarily/\.states\._enableTemporarily/g' \
 -e 's/\.isStateActive/\.states\.isActive/g' \
 -e "s/\(\w\)[[]\([\"']\)$known_states\2[]]/\1\.\3/g" \
 -e "s/self\._registerState(\([\"']\)$known_states\1\,* */\2 = StateVariable(/g" \
 $@

exit 0

sed -i \
 -e 's/(State)/(Stateful)/g' \
 -e 's/import State/import StateVariable, Stateful/g' \
 -e "s/self\._registerState(\([\"']\)$known_states\1\,* */\2 = StateVariable(/g" \
 $@

exit 0


obtained list of all state variables ever known by
grep '_registerState(' *py `find ../mvpa -iname \*.py` 2>/dev/null| \
grep -v 'def _reg' |  sed -e "s/.*(\([\"']\)\([^ ,]*\)\1.*/\2/g"  | \
grep -v mvpa | sort | uniq | tr '\n' '\|'; echo

(State) Stateful

enableState  enable
enabledStates  enabled
listStates listing
states items
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
