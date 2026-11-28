#pragma once

#include <string>

class DragonflyManager;

// Device-scoped identity.
//
// A device bootstraps once (`POST /api/auth/device`) and receives an opaque
// bearer token. The server stores only `SHA-256(token)` -> `{user_id}` in
// Dragonfly, so a database dump does not hand out usable tokens. The token *is*
// the identity: the server derives `user_id` from it and ignores any
// client-supplied id, which is what stops one device from reading or
// overwriting another device's burnout baseline.
//
// There are no user accounts. Losing the token (app data cleared, reinstall)
// means a new identity and a new baseline; `/api/auth/revoke` deletes it.
class DeviceAuth
{
public:
    explicit DeviceAuth(DragonflyManager *dragonfly);

    // True when the token is known and not expired. Fills `user_id`.
    bool authenticate(const std::string &token, std::string &user_id) const;

    // Creates a new identity. Returns the raw token (never stored server-side)
    // and fills `user_id`. Returns "" on failure.
    std::string issue(std::string &user_id) const;

    // Deletes the token and returns the user_id it belonged to ("" if unknown)
    // so the caller can also delete that device's data.
    std::string revoke(const std::string &token) const;

    static std::string hashToken(const std::string &token);

private:
    DragonflyManager *dragonfly_;
};
