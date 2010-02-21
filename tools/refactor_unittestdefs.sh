#!/bin/bash
# silly script to refactor unittest method names from camelcase
for i in {1..10}; do sed -i -e 's/\(def *test[^(]*\)\([A-Z]\)/\1\L_\2\E/g' test*py; done
sed -i \
-e 's/_p_norm/_pnorm/g' \
-e 's/_a_a_c/_aac/g' \
-e 's/_a_c_c/_acc/g' \
-e 's/_k_n_n/_knn/g' \
-e 's/_a_u_c/_auc/g' \
-e 's/_s_v_m/_svm/g' \
-e 's/_s_v_d/_svd/g' \
-e 's/_w_d_m/_wdm/g' \
-e 's/_r_o_i/_roi/g' \
-e 's/_p_l_r/_plr/g' \
-e 's/_e_n_e_t/_enet/g' \
-e 's/_s_m_l_r/_smlr/g' \
-e 's/_l_a_r_s/_lars/g' \
-e 's/_e_e_p/_eep/g' \
-e 's/_i_f_s/_ifs/g' \
-e 's/_r_f_e/_rfe/g' \
-e 's/_g_l_m_n_e_t/_glmnet/g' \
-e 's/_g_l_m/_glm/g' \
-e 's/_g_n_b/_gnb/g' \
-e 's/_b_v_r_t_c/_bv_rtc/g' \
-e 's/__nones/_nones/g' \
-e 's/_e_v/_ev/g' \
-e 's/_l_f/_lf/g' \
-e 's/_c_r/_cr/g' \
-e 's/_w_p/_wp/g' \
-e 's/_s_g/_sg/g' \
-e 's/_c_v/_cv/g' \
-e 's/_e_r/_er/g' \
-e 's/_m_e_g/_meg/g' \
-e 's/_s_o_m/_som/g' \
-e 's/_s_o_m/_som/g' \
-e 's/pnorm_wpython/pnorm_w_python/g' \
-e 's/_p_p_f/_ppf/g' test*py

sed -i -e 's/glmnet__/glmnet_/g' test*py
