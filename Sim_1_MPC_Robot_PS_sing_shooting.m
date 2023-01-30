classdef Sim_1_MPC_Robot_PS_sing_shooting < matlab.System
    % untitled2 Add summary here
    %
    % This template includes the minimum set of functions required
    % to define a System object with discrete state.

    % Public, tunable properties
    properties

    end

    properties (DiscreteState)

    end

    % Pre-computed constants
    properties (Access = private)
        N
        casadi_solver
        ff
        x0
        xs
        lbx
        ubx
        lbg
        ubg 
    end

    methods (Access = protected)
        function num = getNumInputsImpl(~)
            num = 4;
        end
        function num = getNumOutputsImpl(~)
            num = 2;
        end
        function [dt1, dt2] = getOutputDataTypeImpl(~)
        	dt1 = 'double';
            dt2 = 'double';
        end
        function [dt1, dt2, dt3, dt4] = getInputDataTypeImpl(~)
        	dt1 = 'double';
            dt2 = 'double';
            dt3 = 'double';
            dt4 = 'double';
        end
        function [sz1, sz2] = getOutputSizeImpl(~)
        	sz1 = [1,1];
            sz2 = [1,1];
        end
        function [sz1, sz2, sz3, sz4] = getInputSizeImpl(~)
        	sz1 = [1,1];
            sz2 = [1,1];
            sz3 = [1,1];
            sz4 = [1,1];
        end
        function [cp1, cp2, cp3, cp4] = isInputComplexImpl(~)
        	cp1 = false;
            cp2 = false;
            cp3 = false;
            cp4 = false;
        end
        function [cp1, cp2 ] = isOutputComplexImpl(~)
        	cp1 = false;
            cp2 = false;
        end
        function [fz1, fz2, fz3, fz4] = isInputFixedSizeImpl(~)
        	fz1 = true;
            fz2 = true;
            fz3 = true;
            fz4 = true;
        end
        function [fz1, fz2]= isOutputFixedSizeImpl(~)
        	fz1 = true;
            fz2 = true;
        end
        function setupImpl(robot,~,~,~)
            % Perform one-time calculations, such as computing constants
            addpath('C:\Users\quran\Documents\MATLAB\casadi-windows-matlabR2016a-v3.5.5')
            import casadi.*

            T = 0.2; % sampling time [s]
            robot.N = 20; % prediction horizon
            %rob_diam = 0.3;

            v_max = 0.6; v_min = -v_max;
            omega_max = pi/4; omega_min = -omega_max;

            x = SX.sym('x'); y = SX.sym('y'); theta = SX.sym('theta');
            states = [x;y;theta]; n_states = length(states);

            v = SX.sym('v'); omega = SX.sym('omega');
            controls = [v;omega]; n_controls = length(controls);
            rhs = [v*cos(theta);v*sin(theta);omega]; % system r.h.s

            f = Function('f',{states,controls},{rhs}); % nonlinear mapping function f(x,u)
            U = SX.sym('U',n_controls,robot.N); % Decision variables (controls)
            P = SX.sym('P',n_states + n_states);
            % parameters (which include the initial and the reference state of the robot)

            X = SX.sym('X',n_states,(robot.N+1));
            % A Matrix that represents the states over the optimization problem.

            % compute solution symbolically
            X(:,1) = P(1:3); % initial state
            for k = 1:robot.N
                st = X(:,k);  con = U(:,k);
                f_value  = f(st,con);
                st_next  = st+ (T*f_value);
                X(:,k+1) = st_next;
            end
            % this function to get the optimal trajectory knowing the optimal solution
            robot.ff=Function('ff',{U,P},{X});

            obj = 0; % Objective function
            g = [];  % constraints vector

            Q = zeros(3,3); Q(1,1) = 1;Q(2,2) = 5;Q(3,3) = 0.1; % weighing matrices (states)
            R = zeros(2,2); R(1,1) = 0.5; R(2,2) = 0.05; % weighing matrices (controls)
            % compute objective
            for k=1:robot.N
                st = X(:,k);  con = U(:,k);
                obj = obj+(st-P(4:6))'*Q*(st-P(4:6)) + con'*R*con; % calculate obj
            end

            % compute constraints
            for k = 1:robot.N+1   % box constraints due to the map margins
                g = [g; X(1,k)];   %state x
                g = [g; X(2,k)];   %state y
            end

            % make the decision variables one column vector
            OPT_variables = reshape(U,2*robot.N,1);
            nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', P);

            opts = struct;
            opts.ipopt.max_iter = 100;
            opts.ipopt.print_level =0;%0,3
            opts.print_time = 0;
            opts.ipopt.acceptable_tol =1e-8;
            opts.ipopt.acceptable_obj_change_tol = 1e-6;

            solver = nlpsol('solver', 'ipopt', nlp_prob,opts);


            args = struct;
            % inequality constraints (state constraints)
            args.lbg = -2;  % lower bound of the states x and y
            args.ubg = 2;   % upper bound of the states x and y

            % input constraints
            args.lbx(1:2:2*robot.N-1,1) = v_min; args.lbx(2:2:2*robot.N,1)   = omega_min;
            args.ubx(1:2:2*robot.N-1,1) = v_max; args.ubx(2:2:2*robot.N,1)   = omega_max;

            robot.casadi_solver = solver;
            robot.x0 = [0 ; 0 ; 0.0];
            robot.xs = [1.5 ; 1.5 ; 0];
            robot.lbx = args.lbx;
            robot.ubx = args.ubx;
            robot.lbg = args.lbg;
            robot.ubg = args.ubg;
            
        end

        function [u_cl,w_cl] = stepImpl(robot,x,y,theta,t)
            
            if t==0
                x_0=[0; 0; 0];
            else
                x_0=[x; y; theta];
            end
            
            
            u0 = zeros(robot.N,2);
            %w0 = robot.x0;
            lbw = robot.lbx;
            ubw = robot.ubx;
            solver = robot.casadi_solver;

            args.p   = [x_0;robot.xs]; % set the values of the parameters vector
            w0 = reshape(u0',2*robot.N,1); % initial value of the optimization variables
            %tic
            sol = solver('x0', w0, 'lbx', lbw, 'ubx', ubw,...
                'lbg', robot.lbg, 'ubg', robot.ubg,'p',args.p);
            %toc
            u = reshape(full(sol.x)',2,robot.N)';
            ff_value = robot.ff(u',args.p); % compute OPTIMAL solution TRAJECTORY
            %xx1(:,1:3,mpciter+1)= full(ff_value)';

            u_cl= u(1,1);
            w_cl = u(1,2);

        end

        function resetImpl(robot)
            % Initialize / reset discrete-state properties
        end
    end

    methods (Access = protected, Static)
        
        end
    end

