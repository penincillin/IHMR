mlp_default = \
[


    dict(
        update_params = ['pred_hand_trans'],
        loss_weights = dict(
            joints_2d_loss = 10.0,
            joints_3d_loss = 1000.0,
            mano_pose_loss = 10.0,
            mano_shape_loss = 10.0,
            hand_trans_loss = 1000.0,
            shape_reg_loss = 0.1,
            shape_residual_loss = 0,
            collision_loss = 1.0,
        ),
        lr = 1e-4,
        lr_decay_type = 'cosine',
        epoch = 2,
        filter_loss = [
            ('joints_3d_loss_p', '+0'),
            ('collision_loss', '+0'),
        ],
        select_loss = 'collision_loss',
    ),

    dict(
        update_params = ['pred_left_orient'],
        loss_weights = dict(
            joints_2d_loss = 10.0,
            joints_3d_loss = 10.0,
            mano_pose_loss = 10.0,
            mano_shape_loss = 10.0,
            hand_trans_loss = 10.0,
            shape_reg_loss = 0.1,
            collision_loss = 1.0,
            shape_residual_loss = 0.0,
        ),
        lr = 1e-4,
        lr_decay_type = 'cosine',
        epoch = 2,
        filter_loss = [
            ('joints_3d_loss_p', '+0'),
            ('collision_loss', '+0'),
        ],
        select_loss = 'collision_loss',
    ),

    # stage-2
    dict(
        update_params = ['pred_right_orient'],
        loss_weights = dict(
            joints_2d_loss = 10.0,
            joints_3d_loss = 10.0,
            mano_pose_loss = 10.0,
            mano_shape_loss = 10.0,
            hand_trans_loss = 10.0,
            shape_reg_loss = 0.1,
            collision_loss = 1.0,
            shape_residual_loss = 0.0,
        ),
        lr = 1e-4,
        lr_decay_type = 'cosine',
        epoch = 2,
         filter_loss = [
            ('joints_3d_loss_p', '+0'),
            ('collision_loss', '+0'),
        ],
        select_loss = 'collision_loss',
    ),


    dict(
        update_params = ['pred_left_pose_params', 'pred_right_pose_params'],
        loss_weights = dict(
            joints_2d_loss = 10.0,
            joints_3d_loss = 10.0,
            mano_pose_loss = 10.0,
            mano_shape_loss = 10.0,
            hand_trans_loss = 10.0,
            shape_reg_loss = 0.1,
            collision_loss = 1.0,
            shape_residual_loss = 0.0,
        ),
        lr = 1e-4,
        lr_decay_type = 'cosine',
        epoch = 2,
        filter_loss = [
            ('joints_3d_loss_p', '+0'),
            ('collision_loss', '+0'),
        ],
        select_loss = 'collision_loss',
    ),

    dict(
        update_params = ['pred_left_shape_params', 'pred_right_shape_params'],
        loss_weights = dict(
            joints_2d_loss = 10.0,
            joints_3d_loss = 10.0,
            mano_pose_loss = 10.0,
            mano_shape_loss = 10.0,
            hand_trans_loss = 10.0,
            shape_reg_loss = 0.1,
            collision_loss = 1.0,
            shape_residual_loss = 0.0,
        ),
        lr = 1e-4,
        lr_decay_type = 'cosine',
        epoch = 2,
        filter_loss = [
            ('joints_3d_loss_p', '+0'),
            ('collision_loss', '+0'),
        ],
        select_loss = 'collision_loss',
    ),

    # stage-4
    dict(
        update_params = ['pred_cam_params'],
        loss_weights = dict(
            joints_2d_loss = 10.0,
            joints_3d_loss = 10.0,
            mano_pose_loss = 10.0,
            mano_shape_loss = 10.0,
            hand_trans_loss = 10.0,
            shape_reg_loss = 0.1,
            collision_loss = 1.0,
            shape_residual_loss = 0.0,
        ),
        lr = 1e-4,
        lr_decay_type = 'cosine',
        epoch = 5,
        filter_loss = [
            ('joints_2d_loss_p', '+0'),
        ],
        select_loss = 'joints_2d_loss_p',
    ),
]