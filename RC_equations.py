# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 09:57:01 2021

@author: mascherbauer
"""

 for t in timesteps:  # t is the index for each timestep
        # Equ. C.2
        PHI_m = Am / Atot * (0.5 * Q_InternalGains + Q_solar[t, :])
        # Equ. C.3
        PHI_st = (1 - Am / Atot - self.Htr_w / 9.1 / Atot) * \
                 (0.5 * Q_InternalGains + Q_solar[t, :])

        # (T_sup = T_outside weil die Zuluft nicht vorgewärmt oder vorgekühlt wird)
        T_sup[t] = T_outside[t]

        # Equ. C.5
        PHI_mtot_0 = PHI_m + Htr_em * T_outside[t] + Htr_3 * (
                PHI_st + self.Htr_w * T_outside[t] + Htr_1 * (((PHI_ia + 0) / self.Hve) + T_sup[t])) / \
                     Htr_2

        # Equ. C.5 with 10 W/m^2 heating power
        PHI_mtot_10 = PHI_m + Htr_em * T_outside[t] + Htr_3 * (
                PHI_st + self.Htr_w * T_outside[t] + Htr_1 * (
                ((PHI_ia + heating_power_10) / self.Hve) + T_sup[t])) / Htr_2

        # Equ. C.5 with 10 W/m^2 cooling power
        PHI_mtot_10_c = PHI_m + Htr_em * T_outside[t] + Htr_3 * (
                PHI_st + self.Htr_w * T_outside[t] + Htr_1 * (
                ((PHI_ia - heating_power_10) / self.Hve) + T_sup[t])) / Htr_2

        if t == 0:
            if type(initial_thermal_mass_temp) == int or type(initial_thermal_mass_temp) == float:
                Tm_t_prev = np.array([initial_thermal_mass_temp] * len(self.Hve))
            else:  # initial temperature is already a vector
                Tm_t_prev = initial_thermal_mass_temp
        else:
            Tm_t_prev = Tm_t[t - 1, :]

        # Equ. C.4
        Tm_t_0 = (Tm_t_prev * (Cm / 3600 - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot_0) / \
                 (Cm / 3600 + 0.5 * (Htr_3 + Htr_em))

        # Equ. C.4 for 10 W/m^2 heating
        Tm_t_10 = (Tm_t_prev * (Cm / 3600 - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot_10) / \
                  (Cm / 3600 + 0.5 * (Htr_3 + Htr_em))

        # Equ. C.4 for 10 W/m^2 cooling
        Tm_t_10_c = (Tm_t_prev * (Cm / 3600 - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot_10_c) / \
                    (Cm / 3600 + 0.5 * (Htr_3 + Htr_em))

        # Equ. C.9
        T_m_0 = (Tm_t_0 + Tm_t_prev) / 2

        # Equ. C.9 for 10 W/m^2 heating
        T_m_10 = (Tm_t_10 + Tm_t_prev) / 2

        # Equ. C.9 for 10 W/m^2 cooling
        T_m_10_c = (Tm_t_10_c + Tm_t_prev) / 2

        # Euq. C.10
        T_s_0 = (Htr_ms * T_m_0 + PHI_st + self.Htr_w * T_outside[t] + Htr_1 *
                 (T_sup[t] + (PHI_ia + 0) / self.Hve)) / (Htr_ms + self.Htr_w + Htr_1)

        # Euq. C.10 for 10 W/m^2 heating
        T_s_10 = (Htr_ms * T_m_10 + PHI_st + self.Htr_w * T_outside[t] + Htr_1 *
                  (T_sup[t] + (PHI_ia + heating_power_10) / self.Hve)) / (Htr_ms + self.Htr_w + Htr_1)

        # Euq. C.10 for 10 W/m^2 cooling
        T_s_10_c = (Htr_ms * T_m_10_c + PHI_st + self.Htr_w * T_outside[t] + Htr_1 *
                    (T_sup[t] + (PHI_ia - heating_power_10) / self.Hve)) / (Htr_ms + self.Htr_w + Htr_1)

        # Equ. C.11
        T_air_0 = (Htr_is * T_s_0 + self.Hve * T_sup[t] + PHI_ia + 0) / \
                  (Htr_is + self.Hve)

        # Equ. C.11 for 10 W/m^2 heating
        T_air_10 = (Htr_is * T_s_10 + self.Hve * T_sup[t] + PHI_ia + heating_power_10) / \
                   (Htr_is + self.Hve)

        # Equ. C.11 for 10 W/m^2 cooling
        T_air_10_c = (Htr_is * T_s_10_c + self.Hve * T_sup[t] + PHI_ia - heating_power_10) / \
                     (Htr_is + self.Hve)

        for i in range(len(self.Hve)):
            # Check if air temperature without heating is in between boundaries and calculate actual HC power:
            if T_air_0[i] >= T_air_min[t] and T_air_0[i] <= T_air_max[t]:
                Q_Heating_noDR[t, i] = 0
            elif T_air_0[i] < T_air_min[t]:  # heating is required
                Q_Heating_noDR[t, i] = heating_power_10[i] * (T_air_min[t] - T_air_0[i]) / (T_air_10[i] - T_air_0[i])
            elif T_air_0[i] > T_air_max[t]:  # cooling is required
                Q_Cooling_noDR[t, i] = heating_power_10[i] * (T_air_max[t] - T_air_0[i]) / (T_air_10_c[i] - T_air_0[i])

        # now calculate the actual temperature of thermal mass Tm_t with Q_HC_real:
        # Equ. C.5 with actual heating power
        PHI_mtot_real = PHI_m + Htr_em * T_outside[t] + Htr_3 * (
                PHI_st + self.Htr_w * T_outside[t] + Htr_1 * (
                ((PHI_ia + Q_Heating_noDR[t, :] - Q_Cooling_noDR[t, :]) / self.Hve) + T_sup[t])) / Htr_2
        # Equ. C.4
        Tm_t[t, :] = (Tm_t_prev * (Cm / 3600 - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot_real) / \
                     (Cm / 3600 + 0.5 * (Htr_3 + Htr_em))

        # Equ. C.9
        T_m_real = (Tm_t[t, :] + Tm_t_prev) / 2

        # Euq. C.10
        T_s_real = (Htr_ms * T_m_real + PHI_st + self.Htr_w * T_outside[t] + Htr_1 *
                    (T_sup[t] + (PHI_ia + Q_Heating_noDR[t, :] - Q_Cooling_noDR[t, :]) / self.Hve)) / \
                   (Htr_ms + self.Htr_w + Htr_1)

        # Equ. C.11 for 10 W/m^2 heating
        T_Room_noDR[t, :] = (Htr_is * T_s_real + self.Hve * T_sup[t] + PHI_ia +
                             Q_Heating_noDR[t, :] - Q_Cooling_noDR[t, :]) / (Htr_is + self.Hve)

