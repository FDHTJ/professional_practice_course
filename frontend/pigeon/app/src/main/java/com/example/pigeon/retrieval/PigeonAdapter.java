package com.example.pigeon.retrieval;

import android.content.Context;
import android.content.Intent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.example.pigeon.R;
import com.example.pigeon.utills.Pigeon;

import java.util.ArrayList;

public class PigeonAdapter extends RecyclerView.Adapter<PigeonAdapter.ViewHolder> {

    ArrayList<Pigeon> pigeons;
    Context context;

    public PigeonAdapter(ArrayList<Pigeon> pigeons, Context context) {
        this.pigeons = pigeons;
        this.context = context;
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(context).inflate(R.layout.retrieval_result_item, parent, false);
        return new ViewHolder(view);
    }



    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        Pigeon p = pigeons.get(position);

        holder.name.setText(p.name);
        holder.partOfDetails.setText(p.details.substring(0,Math.min(20,p.details.length()))+"...");
        // 加载图片（推荐用 Glide）
        Glide.with(context)
                .load(p.image)
                .error(R.drawable.no_image)      // ✅ 重点：如果图片加载失败（404/断网），显示这张图
                .fallback(R.drawable.no_image)
                .into(holder.image);


        // 点击跳转详情页
        holder.itemView.setOnClickListener(v -> {
            Intent intent = new Intent(context, RetrievalDetails.class);
            intent.putExtra("pigeon", p);
            context.startActivity(intent);
        });
    }

    @Override
    public int getItemCount() {
        return pigeons.size();
    }

    public void addMore(ArrayList<Pigeon> newItems) {
        int start = pigeons.size();
        pigeons.addAll(newItems);
        notifyItemRangeInserted(start, newItems.size());
    }

    public class ViewHolder extends RecyclerView.ViewHolder {
        ImageView image;
        TextView name, partOfDetails;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);

            image = itemView.findViewById(R.id.retrieval_item_image);
            name = itemView.findViewById(R.id.retrieval_item_name);
            partOfDetails = itemView.findViewById(R.id.retrieval_item_introduction);
        }
    }
}
