CREATE TABLE yrno_nvchb2(id INT AUTO_INCREMENT UNIQUE,
					fcst_id VARCHAR(20),
					DateTime VARCHAR(20),
                    air_pressure double,
                    air_temperature double,
                    cloud_area_frac double,
                    cloud_area_frac_high double,
                    cloud_area_frac_medium double,
                    cloud_area_frac_low double,
                    dew_point_temp double,
                    fog_area_frac double,
                    rel_humidity double,
                    uv_idx_clear_sky double,
                    wind_speed double,
                    wind_dir double
                    );
					