from flask import render_template, request, jsonify


def register_error_handlers(app):
    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith('/api/'):
            return jsonify(error="Not found"), 404
        return render_template('error.html', code=404, message="Page not found"), 404

    @app.errorhandler(429)
    def rate_limited(e):
        if request.path.startswith('/api/'):
            return jsonify(error="Rate limit exceeded. Please wait before trying again."), 429
        return render_template('error.html', code=429, message="Too many requests. Please slow down."), 429

    @app.errorhandler(500)
    def server_error(e):
        if request.path.startswith('/api/'):
            return jsonify(error="Internal server error"), 500
        return render_template('error.html', code=500, message="Something went wrong"), 500
