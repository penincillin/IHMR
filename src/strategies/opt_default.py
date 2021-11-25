opt_default = \
[

    dict(
        update_params = ['pred_hand_trans'],
        loss_weights = dict(
            joints_2d_loss = 100.0, 
            joints_3d_loss = 1000.0,
            trans_loss_weight = 1000.0, 
            shape_reg_loss_weight = 0.1,
            collision_loss_weight = 0.1, 
            finger_reg_loss_weight = 0.0,
        ),
        lr = 1e-4,
        epoch = 300,
        filter_loss = [
            ('joints_3d_loss_p', '+0'),
            ('collision_loss', '-10'),
        ],
        select_loss = 'joints_3d_loss_p',
    ),
 
    dict(
        update_params = ['pred_left_orient', 'pred_right_orient'],
        loss_weights = dict(
            joints_2d_loss = 10.0,
            joints_3d_loss = 1000.0,
            trans_loss_weight = 100.0,
            shape_reg_loss_weight = 0.1,
            collision_loss_weight = 1.0,
            finger_reg_loss_weight = 0.0,
        ),
        lr = 1e-2,
        epoch = 300,
        filter_loss = [
            ('joints_3d_loss_p', '+0'),
            ('collision_loss', '-10'),
        ],
        select_loss = 'joints_3d_loss_p',
    ),

    dict(
        update_params = ['pred_left_pose_params', 'pred_right_pose_params'],
        loss_weights = dict(
            joints_2d_loss = 10.0,
            joints_3d_loss = 1000.0,
            trans_loss_weight = 100.0,
            shape_reg_loss_weight = 0.1,
            collision_loss_weight = 1.0,
            finger_reg_loss_weight = 100000.0,
        ),
        lr = 1e-2,
        epoch = 300,
        filter_loss = [
            ('joints_3d_loss_p', '+0'),
            ('collision_loss', '-10'),
        ],
        select_loss = 'joints_3d_loss_p',
    ),

    dict(
        update_params = ['pred_left_shape_params', 'pred_right_shape_params'],
        loss_weights = dict(
            joints_2d_loss = 10.0,
            joints_3d_loss = 1000.0,
            trans_loss_weight = 100.0,
            shape_reg_loss_weight = 0.1,
            collision_loss_weight = 1.0,
            finger_reg_loss_weight = 0.0,
        ),
        lr = 1e-2,
        epoch = 300,
        filter_loss = [
            ('joints_3d_loss_p', '+0'),
            ('collision_loss', '-10'),
        ],
        select_loss = 'joints_3d_loss_p',
    ),


    # dict(
    #     update_params = ['pred_cam_params'],
    #     loss_weights = dict(
    #         joints_2d_loss = 10.0,
    #         joints_3d_loss = 1000.0,
    #         trans_loss_weight = 100.0,
    #         shape_reg_loss_weight = 0.01,
    #         collision_loss_weight = 1.0,
    #         finger_reg_loss_weight = 0.0,
    #     ),
    #     lr = 1e-2,
    #     epoch = 100,
    #     filter_loss = [
    #         ('joints_2d_loss_p', '+0'),
    #     ],
    #     select_loss = 'joints_2d_loss_p',
    # ),
]