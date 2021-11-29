FROM mcr.microsoft.com/azure-functions/dotnet:3.0 AS base
WORKDIR /home/site/wwwroot
EXPOSE 80

FROM mcr.microsoft.com/dotnet/sdk:3.1 AS build
WORKDIR /src
COPY ["OnnxConsume/OnnxConsume.csproj", "OnnxConsume/"]
RUN dotnet restore "OnnxConsume/OnnxConsume.csproj"
COPY . .
WORKDIR "/src/OnnxConsume"
RUN dotnet build "OnnxConsume.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "OnnxConsume.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /home/site/wwwroot
COPY --from=publish /app/publish .
ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true