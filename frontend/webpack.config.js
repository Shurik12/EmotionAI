const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyWebpackPlugin = require('copy-webpack-plugin');

module.exports = {
	entry: './src/index.js',
	output: {
		path: path.resolve(__dirname, 'build'),
		filename: 'bundle.js',
		publicPath: '/'
	},
	module: {
		rules: [
			{
				test: /\.(js|jsx)$/,
				exclude: /node_modules/,
				use: {
					loader: 'babel-loader'
				}
			},
			{
				test: /\.css$/,
				use: ['style-loader', 'css-loader']
			},
			{
				test: /\.(png|svg|jpg|jpeg|gif|webp)$/i,
				type: 'asset',
				generator: {
					filename: 'static/media/[name][ext]'
				},
				parser: {
					dataUrlCondition: {
						maxSize: 8 * 1024 // 8kb - files smaller than this will be inlined as base64
					}
				}
			},
		]
	},
	plugins: [
		// HtmlWebpackPlugin for main index.html
		new HtmlWebpackPlugin({
			template: 'public/index.html',
			filename: 'index.html'
		}),
		// CopyWebpackPlugin for additional files
		new CopyWebpackPlugin({
			patterns: [
				{
					from: 'public/static/privacy-policy.html',
					to: 'privacy-policy.html'
				},
				{
					from: 'public/favicon.ico',
					to: 'favicon.ico'
				}
			]
		})
	],
	devServer: {
		historyApiFallback: true,
		static: {
			directory: path.join(__dirname, 'public')
		},
		port: 3000
	},
	resolve: {
		extensions: ['.js', '.jsx']
	}
};