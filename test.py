import torch
class ActFn(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, alpha):
		ctx.save_for_backward(x, alpha)
		# y_1 = 0.5 * ( torch.abs(x).detach() - torch.abs(x - alpha).detach() + alpha.item() )
		y = torch.clamp(x, min = 0, max = alpha.item())
		scale = (2** - 1) / alpha
		y_q = torch.round( y * scale) / scale
		return y_q

	@staticmethod
	def backward(ctx, dLdy_q):
		# Backward function, I borrowed code from
		# https://github.com/obilaniu/GradOverride/blob/master/functional.py
		# We get dL / dy_q as a gradient
		x, alpha, = ctx.saved_tensors
		# Weight gradient is only valid when [0, alpha]
		# Actual gradient for alpha,
		# By applying Chain Rule, we get dL / dy_q * dy_q / dy * dy / dalpha
		# dL / dy_q = argument,  dy_q / dy * dy / dalpha = 0, 1 with x value range
		lower_bound      = x < 0
		upper_bound      = x > alpha
		# x_range       = 1.0-lower_bound-upper_bound
		x_range = ~(lower_bound|upper_bound)
		grad_alpha = torch.sum(dLdy_q * torch.ge(x, alpha).float()).view(-1)
		return dLdy_q * x_range.float(), grad_alpha, None
