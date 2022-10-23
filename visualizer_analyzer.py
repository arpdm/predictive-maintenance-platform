
    def plot_df_single_color(self, data, variables, labels, size=12, labelsize=17, name=None):
        """
        Generic utility function to generate plots on variaus data frames for dataset
        Note: This function is taken from sample notebook provided by the dataset itself.
        """

        plt.clf()
        input_dim = len(variables)
        cols = min(np.floor(input_dim**0.5).astype(int), 4)
        rows = (np.ceil(input_dim / cols)).astype(int)
        gs = gridspec.GridSpec(rows, cols)
        fig = plt.figure(figsize=(size, max(size, rows * 2)))

        for n in range(input_dim):
            ax = fig.add_subplot(gs[n])
            ax.plot(data[variables[n]], marker=".", markerfacecolor="none", alpha=0.7)
            ax.tick_params(axis="x", labelsize=labelsize)
            ax.tick_params(axis="y", labelsize=labelsize)
            plt.ylabel(labels[n], fontsize=labelsize)
            plt.xlabel("Time [s]", fontsize=labelsize)

        plt.tight_layout()

        if name is not None:
            plt.savefig(name, format="png", dpi=300)

        plt.show()
        plt.close()

    def plot_df_color_per_unit(self, data, variables, labels, size=7, labelsize=17, option="Time", name=None):
        """
        Generic utility function to generate plots on variaus data frames overlayed in one plot
        Note: This function is taken from sample notebook provided by the dataset itself.
        """

        plt.clf()
        input_dim = len(variables)
        cols = min(np.floor(input_dim**0.5).astype(int), 4)
        rows = (np.ceil(input_dim / cols)).astype(int)
        gs = gridspec.GridSpec(rows, cols)
        leg = []
        fig = plt.figure(figsize=(size, max(size, rows * 2)))

        unit_sel = np.unique(data["unit"])
        for n in range(input_dim):
            ax = fig.add_subplot(gs[n])
            for j in unit_sel:
                data_unit = data.loc[data["unit"] == j]
                if option == "cycle":
                    time_s = data.loc[data["unit"] == j, "cycle"]
                    label_x = "Time [cycle]"
                else:
                    time_s = np.arange(len(data_unit))
                    label_x = "Time [s]"
                ax.plot(
                    time_s,
                    data_unit[variables[n]],
                    "-o",
                    color=COLOR_DIC_UNIT["Unit " + str(int(j))],
                    alpha=0.7,
                    markersize=5,
                )
                ax.tick_params(axis="x", labelsize=labelsize)
                ax.tick_params(axis="y", labelsize=labelsize)
                leg.append("Unit " + str(int(j)))
            plt.ylabel(labels[n], fontsize=labelsize)
            plt.xlabel(label_x, fontsize=labelsize)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ",")))
            if n == 0:
                ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ",")))
        plt.legend(leg, loc="best", fontsize=labelsize - 2)  # lower left
        plt.tight_layout()

        if name is not None:
            plt.savefig(name, format="png", dpi=300)

        plt.show()
        plt.close()

    def plot_kde(self, leg, variables, labels, size, units, df_W, df_A, labelsize=17, name=None):
        """
        Flight conditions histogram.
        Note: This function is taken from sample notebook provided by the dataset itself.
        """

        plt.clf()
        input_dim = len(variables)
        cols = min(np.floor(input_dim**0.5).astype(int), 4)
        rows = (np.ceil(input_dim / cols)).astype(int)
        gs = gridspec.GridSpec(rows, cols)
        fig = plt.figure(figsize=(size, max(size, rows * 2)))

        for n in range(input_dim):
            ax = fig.add_subplot(gs[n])
            for k, elem in enumerate(units):
                sns.kdeplot(
                    df_W.loc[df_A["unit"] == elem, variables[n]], color=COLOR_DIC_UNIT[leg[k]], shade=True, gridsize=100
                )
                ax.tick_params(axis="x", labelsize=labelsize)
                ax.tick_params(axis="y", labelsize=labelsize)

            ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ",")))
            plt.xlabel(labels[n], fontsize=labelsize)
            plt.ylabel("Density [-]", fontsize=labelsize)
            if n == 0:
                plt.legend(leg, fontsize=labelsize - 4, loc=0)
            else:
                plt.legend(leg, fontsize=labelsize - 4, loc=2)
        plt.tight_layout()
        if name is not None:
            plt.savefig(name, format="png", dpi=300)
        plt.show()
        plt.close()

    def generate_sensor_readings_graphs_single_unit(self, engine_unit):
        """
        Generate graphs for all sensor readings on one engine unit
        """

        df_x_s_u = self.df_x_s.loc[self.df_aux.unit == engine_unit]
        df_x_s_u.reset_index(inplace=True, drop=True)
        labels = self.x_s_var_names
        self.plot_df_single_color(df_x_s_u, self.x_s_var_names, labels)

    def generate_sensor_readings_graphs_single_unit_single_cycle(self, engine_unit, cycle):
        """
        Generate graphs for all sensor readings on one engine unit at a single specified cycle
        """

        df_x_s_u_c = self.df_x_s.loc[(self.df_aux.unit == engine_unit) & (self.df_aux.cycle == cycle)]
        df_x_s_u_c.reset_index(inplace=True, drop=True)
        self.plot_df_single_color(df_x_s_u_c, self.x_s_var_names, self.x_s_var_names)

    def plot_health_states_for_all_engines(self):
        self.plot_df_color_per_unit(self.df_aux, ["hs"], [r"$h_s$ [-]"], option="cycle")

    def generate_flight_profle_single_unit_single_cycle(self, engine_unit, cycle):
        """
        Generate flight profile for one unit based on scenario-descriptor operating conditions
        """

        df_W_u = self.df_w.loc[(self.df_aux.unit == engine_unit) & (self.df_aux.cycle == cycle)]
        df_W_u.reset_index(inplace=True, drop=True)
        labels = [
            "Altitude [ft]",
            "Mach Number [-]",
            "Throttle Resolver Angle [%]",
            "Temperature at fan inlet (T2) [°R]",
        ]
        self.plot_df_color_per_unit(
            df_W_u, self.w_var_names, labels, size=12, labelsize=19, name=f"flight_profile_{engine_unit}.png"
        )

    def generate_flight_envelope(self):
        """
        Generate flight evelope for each engine class.
        Note: This function is taken from sample notebook provided by the dataset itself.
        """

        labelsize = 17
        x = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
        u = np.array([1.7, 1.7, 4.0, 4.0, 4.0]) * 10000
        l = np.array([0.0, 0.0, 0.0, 0.0, 1.0]) * 10000

        plt.figure(figsize=(7, 5))
        plt.fill_between(x, l, u, alpha=0.2)

        plt.plot(
            self.df_w.loc[self.df_aux["Fc"] == 3, "Mach"], self.df_w.loc[self.df_aux["Fc"] == 3, "alt"], ".", alpha=0.9
        )
        plt.plot(
            self.df_w.loc[self.df_aux["Fc"] == 2, "Mach"], self.df_w.loc[self.df_aux["Fc"] == 2, "alt"], ".", alpha=0.9
        )
        plt.plot(
            self.df_w.loc[self.df_aux["Fc"] == 1, "Mach"], self.df_w.loc[self.df_aux["Fc"] == 1, "alt"], ".", alpha=0.9
        )

        plt.tick_params(axis="x", labelsize=labelsize)
        plt.tick_params(axis="y", labelsize=labelsize)
        plt.xlim((0.0, 0.8))
        plt.ylim((0, 40000))
        plt.xlabel("Mach Number - [-]", fontsize=labelsize)
        plt.ylabel("Flight Altitude - [ft]", fontsize=labelsize)
        plt.show()

    def generate_kde_estimations_of_flight_profile(self):
        """
        Kernel density estimations of the simulated flight envelopes given by recordings of altitude, flight Mach
        number, throttle-resolver angle (TRA) and total temperature at the fan inlet (T2).
        """

        variables = ["alt", "Mach", "TRA", "T2"]
        labels = ["Altitude [ft]", "Mach Number [-]", "Throttle Resolver Angle [%]", "Temperature at fan inlet [°R]"]
        size = 10
        units = list(np.unique(self.df_aux["unit"]))
        leg = ["Unit " + str(int(u)) for u in units]
        self.plot_kde(leg, variables, labels, size, units, self.df_w, self.df_aux, labelsize=19)

    def generate_engine_health_parameter_graphs(self):
        labels = self.t_var_names
        self.plot_df_single_color(self.df_t, self.t_var_names, labels)

    def generate_hpt_eff_over_cycles_all_engines(self):
        self.plot_df_color_per_unit(self.df_t, ["HPT_eff_mod"], [r"HPT Eff. - $\theta$ [-]"], size=7, option="cycle")

    def plot_training_results_history(self, history):
        """
        Takes in history generated by tensorflow when training the model
        and plots the training loss and validation loss against each epoch during training.
        This is a way for us to measure the validity and efficiency of the model.
        """

        plt.figure(figsize=(16, 9))
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train loss", "validation loss"])
        plt.show()
