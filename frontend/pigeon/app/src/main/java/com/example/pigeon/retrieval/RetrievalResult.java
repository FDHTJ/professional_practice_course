package com.example.pigeon.retrieval;

import android.os.Bundle;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.pigeon.R;
import com.example.pigeon.utills.Pigeon;

import java.util.ArrayList;

public class RetrievalResult extends AppCompatActivity {
    ArrayList<Pigeon> pigeons;

    RecyclerView recyclerView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.retrieval_result);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.retrieval_result_layout), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
        pigeons = (ArrayList<Pigeon>) getIntent().getSerializableExtra("pigeonList");

//        System.out.println(pigeons.get(0).details);
        recyclerView = findViewById(R.id.retrieval_recycle_view);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        PigeonAdapter adapter = new PigeonAdapter(pigeons, this);

        recyclerView.setAdapter(adapter);

    }
}