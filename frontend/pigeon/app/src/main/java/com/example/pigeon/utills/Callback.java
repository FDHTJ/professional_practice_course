package com.example.pigeon.utills;

import org.json.JSONException;

public interface Callback {
    void onSuccess(String result) ;
    void onError(Exception e);
}
